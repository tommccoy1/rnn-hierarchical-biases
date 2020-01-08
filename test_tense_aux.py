
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


# Start-of-sentence and end-of-sentence tokens
# The standard seq2seq version only has one EOS. This version has 
# 2 EOS--one signalling that the original sentence should be returned,
# the other signalling it should be reversed.
# I use a 1-hot encoding for all tokens.
SOS_token = 0
EOS_tokenA = 1 # For DECL
EOS_tokenB = 2 # For QUEST



prefix = sys.argv[1] # This means we're using the language with agreement
directory = sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + "_" + sys.argv[6]

counter = 0
dir_made = 0
# Reading the training data
trainingFile = prefix + '.train'
testFile = prefix + '.test'
devFile = prefix + '.dev'
genFile = prefix + '.gen'


batch_size = 5
if "TREE" in sys.argv[3]:
    batch_size = 1

MAX_LENGTH = 20

use_cuda = torch.cuda.is_available()

if use_cuda:
        available_device = torch.device('cuda')
else:
        available_device = torch.device('cpu')


auxes = ["can", "could", "will", "would", "do", "does", "don't", "doesn't"]

word2index = {}
index2word = {}

if sys.argv[1] == "agr_main_tense" or sys.argv[1] == "agr_tense_subject":
        fi = open("index_both.txt", "r")
elif "tense_aux" in sys.argv[1]:
        fi = open("index_tense_aux.txt", "r")
elif "tense" in sys.argv[1]:
        fi = open("index_tense.txt", "r")
else:
        fi = open("index.txt", "r")



for line in fi:
        parts = line.strip().split("\t")
        word2index[parts[0]] = int(parts[1])
        index2word[int(parts[1])] = parts[0]

#word2index["SOS"] = 0
#word2index["."] = 1
#word2index["?"] = 2
#index2word[0] = "SOS"
#index2word[1] = "."
#index2word[2] = "?"

MAX_LENGTH = 20


#train_batches, MAX_LENGTH = file_to_batches(trainingFile, MAX_LENGTH)
#dev_batches, MAX_LENGTH = file_to_batches(devFile, MAX_LENGTH)
test_batches, MAX_LENGTH = file_to_batches(testFile, MAX_LENGTH)
gen_batches, MAX_LENGTH = file_to_batches(genFile, MAX_LENGTH)

print(index2word)

recurrent_unit = sys.argv[3] # Could be "SRN" or "LSTM" instead
attention = sys.argv[4]# Could be "n" instead

if attention == "0":
        attention = 0
elif attention == "1":
        attention = 1
elif attention == "2":
	attention = 2
else:
        print("Please specify 'y' for attention or 'n' for no attention.")


MAX_EXAMPLE = 10000
MAX_LENGTH = 20

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


# Where the actual running of the code happens
hidden_size = int(sys.argv[6]) # Default 128

print(MAX_LENGTH)
print("keys:", len(word2index.keys()))



if recurrent_unit == "TREE":
        encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEENC":
        encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREEDEC":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEBOTH":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREENew":
        encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEENCNew":
        encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREENOPRE":
        encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEENCNOPRE":
        encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREEDECNOPRE":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEBOTHNOPRE":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "TREENewNOPRE":
        encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
        decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
elif recurrent_unit == "TREEENCNewNOPRE":
        encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
elif recurrent_unit == "ONLSTMPROC":
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "ONLSTM", max_length=MAX_LENGTH)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "ONLSTM", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
else:
        encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, recurrent_unit, max_length=MAX_LENGTH)
        decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), recurrent_unit, attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)


encoder1 = encoder1.to(device=available_device)
decoder1 = decoder1.to(device=available_device)



counter = 0
direcs_to_process = 1

allTotalTest = 0
allTestCorrect = 0
allTestCorrectPos = 0
allTotalGen = 0
allGenCorrect = 0
allGenFullsent = 0
allGenFullsentPos = 0
allGenRight = 0
allGenLin = 0
allGenMainRight = 0
allGenMainLin = 0
allGenMainRightnum = 0
allGenMainWrongnum = 0

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


while direcs_to_process:
        if not os.path.exists(directory + "_" +  str(counter)):
                direcs_to_process = 0
        else:
                directory_now = directory + "_" + str(counter)
                counter += 1
		
                dec_list = sorted(os.listdir(directory_now))
                dec = sorted(dec_list[:int(len(dec_list)/2)], key=lambda x:float(".".join(x.split(".")[2:4])))[0]
                print("This directory:", dec)
                enc = dec.replace("decoder", "encoder")


                try:
                        encoder1.load_state_dict(torch.load(directory_now + "/" + enc))
                        decoder1.load_state_dict(torch.load(directory_now + "/" + dec))
                except RuntimeError:   
                        if sys.argv[1] == "agr_main_tense" or sys.argv[1] == "agr_tense_subject":
                                fi = open("index_both.txt", "r") 
                        elif "tense_aux" in sys.argv[1]:
                                fi = open("index_tense_aux.txt", "r")
                        elif len(word2index.keys()) == 81:
                                fi = open("index_tense.txt", "r")
                        else:
                                fi = open("index.txt", "r")
                    
                        word2index = {}
                        index2word = {}


                        #fi = open("index.txt", "r")
                        for line in fi:
                                parts = line.strip().split("\t")
                                word2index[parts[0]] = int(parts[1])
                                index2word[int(parts[1])] = parts[0]

                        if recurrent_unit == "TREE":
                                encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEENC":
                                encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREEDEC":
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEBOTH":
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREENew":
                                encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEENCNew":
                                encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREENOPRE": 
                                encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEENCNOPRE":
                                encoder1 = TreeEncoderRNN(len(word2index.keys()), hidden_size)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREEDECNOPRE":
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEBOTHNOPRE":
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "GRU", max_length=MAX_LENGTH)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "TREENewNOPRE":
                                encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
                                decoder1 = TreeDecoderRNN(len(word2index.keys()), hidden_size)
                        elif recurrent_unit == "TREEENCNewNOPRE":
                                encoder1 = TreeEncoderRNNNew(len(word2index.keys()), hidden_size)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "GRU", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        elif recurrent_unit == "ONLSTMPROC": 
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, "ONLSTM", max_length=MAX_LENGTH)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), "ONLSTM", attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
                        else: 
                                encoder1 = EncoderRNN(len(word2index.keys()), hidden_size, recurrent_unit, max_length=MAX_LENGTH)
                                decoder1 = DecoderRNN(hidden_size, len(word2index.keys()), recurrent_unit, attn=attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)

                        encoder1.load_state_dict(torch.load(directory_now + "/" + enc))
                        decoder1.load_state_dict(torch.load(directory_now + "/" + dec))

                        encoder1 = encoder1.to(device=available_device)
                        decoder1 = decoder1.to(device=available_device)


                        
                print("Test")
                evaluateRandomly(encoder1, decoder1, test_batches, index2word)
                print("Gen")
                evaluateRandomly(encoder1, decoder1, gen_batches, index2word)
                print("Next")
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

                allTestCorrect += right
                allTotalTest += total
                allTestCorrectPos += rightpos
                test_full_sent.append(right * 1.0 / total)
                test_full_sent_pos.append(rightpos * 1.0 / total)

                # Counts for how many sentences in the generalization set was the correct auxiliary predicted
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
                this_gen_main_rightnum = 0
                this_gen_main_wrongnum = 0

                for this_batch in gen_batches:
                        input_sents = logits_to_sentence(this_batch[0], index2word, end_at_punc=False)
                        target_sents = logits_to_sentence(this_batch[1], index2word)
                        pred_sents = logits_to_sentence(evaluate(encoder1, decoder1, this_batch), index2word)


                        for trio in zip(input_sents, target_sents, pred_sents):
                                input_sent = trio[0]
                                target_sent = trio[1]
                                pred_sent = trio[2]

                                correct_words = target_sent.split()#batch[index][1].split()

                                total += 1

                                if pred_sent == target_sent:#batch[index][1]:
                                    full_right += 1
                                    this_gen_right += 1
                                if pred_sent == tense_nearest_aux(target_sent):
                                    this_gen_lin += 1
                               
                                if main_right_tense_aux(target_sent, pred_sent):
                                    this_gen_main_right += 1
                                if main_linear_tense_aux(target_sent, pred_sent):
                                    this_gen_main_lin += 1
                                if main_rightnum_tense_aux(target_sent, pred_sent):
                                    this_gen_main_rightnum += 1
                                if main_wrongnum_tense_aux(target_sent, pred_sent):
                                    this_gen_main_wrongnum += 1
 
                                if sent_to_pos(pred_sent) == sent_to_pos(target_sent):#batch[index][1]):
                                    full_right_pos += 1

                print("Number of sentences with the correct prediction:", right)
                print("Number of sentences fully correct:", full_right)
                print("Total number of sentences", total)
		
                allTotalGen += total
                allGenCorrect += right
               
                allGenRight += this_gen_right
                allGenLin += this_gen_lin
                allGenMainRight += this_gen_main_right
                allGenMainLin += this_gen_main_lin
                allGenMainRightnum += this_gen_main_rightnum
                allGenMainWrongnum += this_gen_main_wrongnum
 
                gen_full_sent.append(full_right * 1.0 / total)
                gen_full_sent_pos.append(full_right_pos * 1.0 / total)
                
                gen_right.append(this_gen_right * 1.0 / total)
                gen_lin.append(this_gen_lin * 1.0 / total)
                gen_main_right.append(this_gen_main_right * 1.0 / total)
                gen_main_lin.append(this_gen_main_lin * 1.0 / total)
                gen_main_rightnum.append(this_gen_main_rightnum * 1.0 / total)
                gen_main_wrongnum.append(this_gen_main_wrongnum * 1.0 / total)

print("Overall test correct:", allTestCorrect)
print("Overall test total:", allTotalTest)
print("Overall test accuracy:", allTestCorrect * 1.0 / allTotalTest)
print("Test accuracy list:")
print(", ".join([str(x) for x in test_full_sent]))
print("Mean:", str(mean(test_full_sent)))
print("Median:", str(median(test_full_sent)))
print("Mean10:", str(mean(test_full_sent[:10])))
print("Median10:", str(median(test_full_sent[:10])))
print(" ")
print("Overall test correct POS:", allTestCorrectPos)
print("Overall test total:", allTotalTest)
print("Overall test accuracy:", allTestCorrectPos * 1.0 / allTotalTest)
print("Test accuracy list:")
print(", ".join([str(x) for x in test_full_sent_pos]))
print("Mean:", str(mean(test_full_sent_pos)))
print("Median:", str(median(test_full_sent_pos)))
print("Mean10:", str(mean(test_full_sent_pos[:10])))
print("Median10:", str(median(test_full_sent_pos[:10])))
print(" ")


print("Overall gen full sentence correct:", allGenFullsent)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenFullsent * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_full_sent]))
print("Mean:", str(mean(gen_full_sent)))
print("Median:", str(median(gen_full_sent)))
print("Mean10:", str(mean(gen_full_sent[:10])))
print("Median10:", str(median(gen_full_sent[:10])))
print(" ")
print("Overall gen full sentence POS correct:", allGenFullsentPos)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenFullsentPos * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_full_sent_pos]))
print("Mean:", str(mean(gen_full_sent_pos)))
print("Median:", str(median(gen_full_sent_pos)))
print("Mean10:", str(mean(gen_full_sent_pos[:10])))
print("Median10:", str(median(gen_full_sent_pos[:10])))
print(" ")



print("Overall gen full sentence right:", allGenRight)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenRight * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_right]))
print("Mean:", str(mean(gen_right)))
print("Median:", str(median(gen_right)))
print("Mean10:", str(mean(gen_right[:10])))
print("Median10:", str(median(gen_right[:10])))
print(" ")
print("Overall gen full linear sentence:", allGenLin)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenLin * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_lin]))
print("Mean:", str(mean(gen_lin)))
print("Median:", str(median(gen_lin)))
print("Mean10:", str(mean(gen_lin[:10])))
print("Median10:", str(median(gen_lin[:10])))
print(" ")
print("Overall gen main verb correct:", allGenMainRight)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenMainRight * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_main_right]))
print("Mean:", str(mean(gen_main_right)))
print("Median:", str(median(gen_main_right)))
print("Mean10:", str(mean(gen_main_right[:10])))
print("Median10:", str(median(gen_main_right[:10])))
print(" ")
print("Overall gen main verb linear:", allGenMainLin)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenMainLin * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_main_lin]))
print("Mean:", str(mean(gen_main_lin)))
print("Median:", str(median(gen_main_lin)))
print("Mean10:", str(mean(gen_main_lin[:10])))
print("Median10:", str(median(gen_main_lin[:10])))
print(" ")
print("Overall gen main verb right num:", allGenMainRightnum)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenMainRightnum * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_main_rightnum]))
print("Mean:", str(mean(gen_main_rightnum)))
print("Median:", str(median(gen_main_rightnum)))
print("Mean10:", str(mean(gen_main_rightnum[:10])))
print("Median10:", str(median(gen_main_rightnum[:10])))
print(" ")
print("Overall gen main verb wrong num:", allGenMainWrongnum)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenMainWrongnum * 1.0 / allTotalGen)
print("Gen full sentence list:")
print(", ".join([str(x) for x in gen_main_wrongnum]))
print("Mean:", str(mean(gen_main_wrongnum)))
print("Median:", str(median(gen_main_wrongnum)))
print("Mean10:", str(mean(gen_main_wrongnum[:10])))
print("Median10:", str(median(gen_main_wrongnum[:10])))
print(" ")




