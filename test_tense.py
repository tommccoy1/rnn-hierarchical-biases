
# This code was adapted from the tutorial "Translation with a Sequence to 
# Sequence Network and Attention" by Sean Robertson. It can be found at the
# following URL:
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# You must have PyTorch installed to run this code.
# You can get it from: http://pytorch.org/


# Imports
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sys
import os

# Functions for tracking time
import time
import math
from numpy import median
from numpy import mean

from seq2seq import pos_to_parse, sent_to_pos, file_to_batches

from evaluation import *
from models import *
from parsing import *
from sent_evals import *

random.seed(7)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", help="encoder type", type=str, default=None)
parser.add_argument("--decoder", help="decoder type", type=str, default=None)
parser.add_argument("--task", help="task", type=str, default=None)
parser.add_argument("--attention", help="attention type", type=str, default=None)
parser.add_argument("--lr", help="learning rate", type=float, default=None)
parser.add_argument("--hs", help="hidden size", type=int, default=None)
parser.add_argument("--seed", help="random seed", type=float, default=None)
args = parser.parse_args()


prefix = args.task
directory = args.task + "_" + args.encoder + "_" + args.decoder  + "_" + args.attention + "_" + str(args.lr) + "_" + str(args.hs)


# Reading the training data
trainingFile = 'data/' + prefix + '.train'
testFile = 'data/' + prefix + '.test'
devFile = 'data/' + prefix + '.dev'
genFile = 'data/' + prefix + '.gen'


# Determine if we are using a GPU
use_cuda = torch.cuda.is_available()

if use_cuda:
    available_device = torch.device('cuda')
else:
    available_device = torch.device('cpu')

# Dictionaries for converting words to indices, and vice versa
word2index = {}
index2word = {}

fi = open("index.txt", "r")

for line in fi:
    parts = line.strip().split("\t")
    word2index[parts[0]] = int(parts[1])
    index2word[int(parts[1])] = parts[0]

MAX_LENGTH = 20

test_batches, MAX_LENGTH = file_to_batches(testFile, MAX_LENGTH, batch_size=1)
gen_batches, MAX_LENGTH = file_to_batches(genFile, MAX_LENGTH, batch_size=1)

# Show the output for a few randomly selected sentences
def evaluateRandomly(encoder, decoder, batches, index2word, n=10):

    batch_size = batches[0][0].size()[1]

    for i in range(math.ceil(n * 1.0 / batch_size)):
        this_batch = random.choice(batches)
        
        input_sents = logits_to_sentence(this_batch[0], index2word, end_at_punc=False)
        target_sents = logits_to_sentence(this_batch[1], index2word)
        pred_sents = logits_to_sentence(evaluate(encoder, decoder, this_batch), index2word)

        for group in zip(input_sents, target_sents, pred_sents):
            print(group[0])
            print(group[1])
            print(group[2])
            print("")

# Initialize the encoder and the decoder
if args.encoder == "Tree":
    encoder = TreeEncoderRNN(len(word2index.keys()), args.hs)
else:
    encoder = EncoderRNN(len(word2index.keys()), args.hs, args.encoder, max_length=MAX_LENGTH)

if args.decoder == "Tree":
    # Note that attention is not implemented for the tree decoder
    decoder = TreeDecoderRNN(len(word2index.keys()), args.hs)
else:
    decoder = DecoderRNN(args.hs, len(word2index.keys()), args.decoder, attn=args.attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)

encoder = encoder.to(device=available_device)
decoder = decoder.to(device=available_device)

# Variables for iterating over directories
counter = 0
direcs_to_process = 1

# Lists for keeping track of the metrics for each model
this_gen_right = 0
this_gen_lin = 0
this_gen_main_right = 0
this_gen_main_lin = 0
this_gen_main_rightnum = 0
this_gen_main_wrongnum = 0
test_full_sent = []
test_full_sent_pos = []
gen_full_sent = []
gen_full_sent_pos = []
gen_right = []
gen_lin = []
gen_main_right = []
gen_main_lin = []
gen_main_rightnum = []
gen_main_wrongnum = []

# Iterate over all re-runs of the same model type that has been specified
while direcs_to_process:
    if not os.path.exists(directory + "_" +  str(counter)):
        direcs_to_process = 0
    else:
        directory_now = directory + "_" + str(counter)
        counter += 1
		
        dec_list = sorted(os.listdir(directory_now))
        dec = sorted(dec_list[:int(len(dec_list)/2)], key=lambda x:float(".".join(x.split(".")[2:4])))[0]
        print("Directory being processed::", dec)
        enc = dec.replace("decoder", "encoder")


        encoder.load_state_dict(torch.load(directory_now + "/" + enc))
        decoder.load_state_dict(torch.load(directory_now + "/" + dec))

                        
        print("Test set example outputs")
        evaluateRandomly(encoder, decoder, test_batches, index2word)
        print("Gen set example outputs")
        evaluateRandomly(encoder, decoder, gen_batches, index2word)
        print("Evaluation of model")

        # Evaluation on the test set
        right = 0
        rightpos = 0
        total = 0

        for this_batch in test_batches:
            input_sents = logits_to_sentence(this_batch[0], index2word, end_at_punc=False)
            target_sents = logits_to_sentence(this_batch[1], index2word)
            pred_sents = logits_to_sentence(evaluate(encoder, decoder, this_batch), index2word)

            for trio in zip(input_sents, target_sents, pred_sents):
                input_sent = trio[0]
                target_sent = trio[1]
                pred_sent = trio[2]

                total += 1
                                
                if pred_sent == target_sent:
                    right += 1
                if sent_to_pos(pred_sent) == sent_to_pos(target_sent):
                    rightpos += 1

			
        print("Test number correct:", right)
        print("Test total:", total)

        test_full_sent.append(right * 1.0 / total)
        test_full_sent_pos.append(rightpos * 1.0 / total)

        # Evaluate on the generalization set
        right = 0
        first_aux = 0
        other_aux = 0
        other_word = 0
        total = 0
        other = 0
        full_right = 0
        full_right_pos = 0
                
        this_gen_right = 0
        this_gen_lin = 0
        this_gen_main_right = 0
        this_gen_main_lin = 0
        this_gen_main_wrongnum = 0
        this_gen_main_rightnum = 0

        for this_batch in gen_batches:
            input_sents = logits_to_sentence(this_batch[0], index2word, end_at_punc=False)
            target_sents = logits_to_sentence(this_batch[1], index2word)
            pred_sents = logits_to_sentence(evaluate(encoder, decoder, this_batch), index2word)

            for trio in zip(input_sents, target_sents, pred_sents):
                input_sent = trio[0]
                target_sent = trio[1]
                pred_sent = trio[2]

                correct_words = target_sent.split()

                total += 1

                if pred_sent == target_sent:
                    full_right += 1
                    this_gen_right += 1
                if pred_sent == tense_nearest(target_sent):
                    this_gen_lin += 1
                               
                if main_right_tense(target_sent, pred_sent):
                    this_gen_main_right += 1
                if main_linear_tense(target_sent, pred_sent):
                    this_gen_main_lin += 1
                if main_rightnum_tense(target_sent, pred_sent):
                    this_gen_main_rightnum += 1
                if main_wrongnum_tense(target_sent, pred_sent):
                    this_gen_main_wrongnum += 1

                if sent_to_pos(pred_sent) == sent_to_pos(target_sent):
                    full_right_pos += 1

        gen_full_sent.append(full_right * 1.0 / total)
        gen_full_sent_pos.append(full_right_pos * 1.0 / total)
                
        gen_right.append(this_gen_right * 1.0 / total)
        gen_lin.append(this_gen_lin * 1.0 / total)
        gen_main_right.append(this_gen_main_right * 1.0 / total)
        gen_main_lin.append(this_gen_main_lin * 1.0 / total)
        gen_main_rightnum.append(this_gen_main_rightnum * 1.0 / total)
        gen_main_wrongnum.append(this_gen_main_wrongnum * 1.0 / total)

print("Test full-sentence accuracy list:")
print(", ".join([str(x) for x in test_full_sent]))
print("Mean:", str(mean(test_full_sent)))
print("Median:", str(median(test_full_sent)))
print(" ")
print("Test full-sentence POS accuracy list:")
print(", ".join([str(x) for x in test_full_sent_pos]))
print("Mean:", str(mean(test_full_sent_pos)))
print("Median:", str(median(test_full_sent_pos)))
print(" ")


print("Gen full-sentence accuracy list:")
print(", ".join([str(x) for x in gen_full_sent]))
print("Mean:", str(mean(gen_full_sent)))
print("Median:", str(median(gen_full_sent)))
print(" ")
print("Gen full-sentence POS accuracy list:")
print(", ".join([str(x) for x in gen_full_sent_pos]))
print("Mean:", str(mean(gen_full_sent_pos)))
print("Median:", str(median(gen_full_sent_pos)))
print(" ")



print("Gen proportion of full-sentence outputs that follow agree-recent:")
print(", ".join([str(x) for x in gen_lin]))
print("Mean:", str(mean(gen_lin)))
print("Median:", str(median(gen_lin)))
print(" ")
print("Gen proportion of outputs that have the correct main verb:")
print(", ".join([str(x) for x in gen_main_right]))
print("Mean:", str(mean(gen_main_right)))
print("Median:", str(median(gen_main_right)))
print(" ")
print("Gen proportion of outputs that have the main verb predicted by agree-recent:")
print(", ".join([str(x) for x in gen_main_lin]))
print("Mean:", str(mean(gen_main_lin)))
print("Median:", str(median(gen_main_lin)))
print(" ")
print("Gen proportion of outputs that have the correct number for the main verb:")
print(", ".join([str(x) for x in gen_main_rightnum]))
print("Mean:", str(mean(gen_main_rightnum)))
print("Median:", str(median(gen_main_rightnum)))
print(" ")
print("Gen proportion of outputs that have the incorrect number for the main verb:")
print(", ".join([str(x) for x in gen_main_wrongnum]))
print("Mean:", str(mean(gen_main_wrongnum)))
print("Median:", str(median(gen_main_wrongnum)))
print(" ")

