
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

from seq2seq import sent_to_pos, file_to_batches

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


# Determine whether we are running it on a GPU
use_cuda = torch.cuda.is_available()

if use_cuda:
        available_device = torch.device('cuda')
else:
        available_device = torch.device('cpu')


auxes = ["can", "could", "will", "would", "do", "does", "don't", "doesn't"]

# Create dictionaries for converting words to numerical
# indices and vice versa
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

encoder1 = encoder1.to(device=available_device)
decoder1 = decoder1.to(device=available_device)


# Variables for iterating over directories
counter = 0
direcs_to_process = 1

# Lists where we track statistics for each model
test_full_sent = []
test_full_sent_pos = []
gen_full_sent = []
gen_full_sent_pos = []
gen_first_word = []
gen_first_word_first_aux = []
gen_first_word_other_aux = []
gen_first_word_other_word = []
d1p1_lst = []
d1p2_lst = []
d1po_lst = []
d2p1_lst = []
d2p2_lst = []
d2po_lst = []
dnp1_lst = []
dnp2_lst = []
dnpo_lst = []
other_lst = []
orc_lst = []
srct_lst = []
srci_lst = []

# Iterate over all re-runs of the same model type that has been specified
while direcs_to_process:
        if not os.path.exists(directory + "_" +  str(counter)):
                direcs_to_process = 0
        else:
                directory_now = directory + "_" + str(counter)
                counter += 1
		
                dec_list = sorted(os.listdir(directory_now))
                dec = sorted(dec_list[:int(len(dec_list)/2)], key=lambda x:float(".".join(x.split(".")[2:4])))[0]
                print("Directory being processed:", dec)
                enc = dec.replace("decoder", "encoder")


                encoder1.load_state_dict(torch.load(directory_now + "/" + enc))
                decoder1.load_state_dict(torch.load(directory_now + "/" + dec))
        
                print("Test set example outputs")
                evaluateRandomly(encoder1, decoder1, test_batches, index2word)
                print("Gen set example outputs")
                evaluateRandomly(encoder1, decoder1, gen_batches, index2word)
                print("Evaluation of model")

                # Evaluate on the test set
                right = 0
                rightpos = 0
                total = 0

                for this_batch in test_batches:
                        input_sents = logits_to_sentence(this_batch[0], index2word, end_at_punc=False)
                        target_sents = logits_to_sentence(this_batch[1], index2word)
                        pred_sents = logits_to_sentence(evaluate(encoder1, decoder1, this_batch), index2word)

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
                this_d1p1 = 0
                this_d1p2 = 0
                this_d1po = 0
                this_d2p1 = 0
                this_d2p2 = 0
                this_d2po = 0
                this_dnp1 = 0
                this_dnp2 = 0
                this_dnpo = 0
                this_other = 0
                this_orc = 0
                this_orc_total = 0
                this_srct = 0
                this_srct_total = 0
                this_srci = 0
                this_srci_total = 0

                for this_batch in gen_batches:
                        input_sents = logits_to_sentence(this_batch[0], index2word, end_at_punc=False)
                        target_sents = logits_to_sentence(this_batch[1], index2word)
                        pred_sents = logits_to_sentence(evaluate(encoder1, decoder1, this_batch), index2word)


                        for trio in zip(input_sents, target_sents, pred_sents):
                                input_sent = trio[0]
                                target_sent = trio[1]
                                pred_sent = trio[2]

                                correct_words = target_sent.split()
                                if not two_agreeing_auxes(target_sent):
                                    break

                                total += 1

                                rc_cat = rc_category(input_sent)
                                if rc_cat == "ORC":
                                    this_orc_total += 1
                                elif rc_cat == "SRC_t":
                                    this_srct_total += 1
                                elif rc_cat == "SRC_i":
                                    this_srci_total += 1
                                                            

                                if pred_sent.split()[0] == target_sent.split()[0]:
                                    right += 1
                                    if rc_cat == "ORC":
                                        this_orc += 1
                                    elif rc_cat == "SRC_t":
                                        this_srct += 1
                                    elif rc_cat == "SRC_i":
                                        this_srci += 1
                                                            

                                elif pred_sent.split()[0] in target_sent.split() and pred_sent.split()[0] in auxes:
                                    first_aux += 1
                                elif pred_sent.split()[0] in auxes:
                                    other_aux += 1
                                else:
                                    other_word += 1

                                if pred_sent == target_sent:
                                    full_right += 1
                                if sent_to_pos(pred_sent) == sent_to_pos(target_sent):
                                    full_right_pos += 1

                                crain_class = crain(input_sent, pred_sent)
                                if crain_class == "d1p1":
                                    this_d1p1 += 1
                                elif crain_class == "d1p2":
                                    this_d1p2 += 1
                                elif crain_class == "d1po":
                                    this_d1po += 1
                                elif crain_class == "d2p1":
                                    this_d2p1 += 1
                                elif crain_class == "d2p2":
                                    this_d2p2 += 1
                                elif crain_class == "d2po":
                                    this_d2po += 1
                                elif crain_class == "dnp1":
                                    this_dnp1 += 1
                                elif crain_class == "dnp2":
                                    this_dnp2 += 1
                                elif crain_class == "dnpo":
                                    this_dnpo += 1
                                else:
                                    this_other += 1

                gen_full_sent.append(full_right * 1.0 / total)
                gen_full_sent_pos.append(full_right_pos * 1.0 / total)
                gen_first_word.append(right * 1.0 / total)
                gen_first_word_first_aux.append(first_aux * 1.0/total)
                gen_first_word_other_aux.append(other_aux * 1.0 / total)
                gen_first_word_other_word.append(other_word * 1.0 / total)

                d1p1_lst.append(this_d1p1 * 1.0/total)
                d1p2_lst.append(this_d1p2 * 1.0/total)
                d1po_lst.append(this_d1po * 1.0/total)
                d2p1_lst.append(this_d2p1 * 1.0/total)
                d2p2_lst.append(this_d2p2 * 1.0/total)
                d2po_lst.append(this_d2po * 1.0/total)
                dnp1_lst.append(this_dnp1 * 1.0/total)
                dnp2_lst.append(this_dnp2 * 1.0/total)
                dnpo_lst.append(this_dnpo * 1.0/total)
                other_lst.append(this_other * 1.0/total)

                orc_lst.append(this_orc * 1.0/this_orc_total)
                srct_lst.append(this_srct * 1.0/this_srct_total)
                srci_lst.append(this_srci * 1.0/this_srci_total)


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

print("Gen first word accuracy list:")
print(", ".join([str(x) for x in gen_first_word]))
print("Mean:", str(mean(gen_first_word)))
print("Median:", str(median(gen_first_word)))
print(" ")
print("Gen proportion of outputs where the first word was the first auxiliary:")
print(", ".join([str(x) for x in gen_first_word_first_aux]))
print("Mean:", str(mean(gen_first_word_first_aux)))
print("Median:", str(median(gen_first_word_first_aux)))
print(" ")
print("Gen proportion of outputs where the first word was an auxiliary not in the input:")
print(", ".join([str(x) for x in gen_first_word_other_aux]))
print("Mean:", str(mean(gen_first_word_other_aux)))
print("Median:", str(median(gen_first_word_other_aux)))
print(" ")
print("Gen proportion of outputs where the first word was not an auxiliary:")
print(", ".join([str(x) for x in gen_first_word_other_word]))
print("Mean:", str(mean(gen_first_word_other_word)))
print("Median:", str(median(gen_first_word_other_word)))
print(" ")


print("Gen full sentence accuracy list:")
print(", ".join([str(x) for x in gen_full_sent]))
print("Mean:", str(mean(gen_full_sent)))
print("Median:", str(median(gen_full_sent)))
print(" ")
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_full_sent_pos]))
print("Mean:", str(mean(gen_full_sent_pos)))
print("Median:", str(median(gen_full_sent_pos)))
print(" ")



print("d1p1 list:")
print(", ".join([str(x) for x in d1p1_lst]))
print("Mean:", str(mean(d1p1_lst)))
print("Median:", str(median(d1p1_lst)))
print(" ")
print("d1p2 list:")
print(", ".join([str(x) for x in d1p2_lst]))
print("Mean:", str(mean(d1p2_lst)))
print("Median:", str(median(d1p2_lst)))
print(" ")
print("d1po list:")
print(", ".join([str(x) for x in d1po_lst]))
print("Mean:", str(mean(d1po_lst)))
print("Median:", str(median(d1po_lst)))
print(" ")
print("d2p1 list:")
print(", ".join([str(x) for x in d2p1_lst]))
print("Mean:", str(mean(d2p1_lst)))
print("Median:", str(median(d2p1_lst)))
print(" ")
print("d2p2 list:")
print(", ".join([str(x) for x in d2p2_lst]))
print("Mean:", str(mean(d2p2_lst)))
print("Median:", str(median(d2p2_lst)))
print(" ")
print("d2po list:")
print(", ".join([str(x) for x in d2po_lst]))
print("Mean:", str(mean(d2po_lst)))
print("Median:", str(median(d2po_lst)))
print(" ")
print("dnp1 list:")
print(", ".join([str(x) for x in dnp1_lst]))
print("Mean:", str(mean(dnp1_lst)))
print("Median:", str(median(dnp1_lst)))
print(" ")
print("dnp2 list:")
print(", ".join([str(x) for x in dnp2_lst]))
print("Mean:", str(mean(dnp2_lst)))
print("Median:", str(median(dnp2_lst)))
print(" ")
print("dnpo list:")
print(", ".join([str(x) for x in dnpo_lst]))
print("Mean:", str(mean(dnpo_lst)))
print("Median:", str(median(dnpo_lst)))
print(" ")
print("other list:")
print(", ".join([str(x) for x in other_lst]))
print("Mean:", str(mean(other_lst)))
print("Median:", str(median(other_lst)))
print("")

print("ORC list:")
print(", ".join([str(x) for x in orc_lst]))
print("Mean:", str(mean(orc_lst)))
print("Median:", str(median(orc_lst)))
print("")

print("SRC_t list:")
print(", ".join([str(x) for x in srct_lst]))
print("Mean:", str(mean(srct_lst)))
print("Median:", str(median(srct_lst)))
print("")

print("SRC_i list:")
print(", ".join([str(x) for x in srci_lst]))
print("Mean:", str(mean(srci_lst)))
print("Median:", str(median(srci_lst)))
print("")





