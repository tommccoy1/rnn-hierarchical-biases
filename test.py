
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
allGenFirstAux = 0
allGenOtherAux = 0
allGenOtherWord = 0

genORCCorrect = 0
genORCTotal = 0
genSRCtCorrect = 0
genSRCtTotal = 0
genSRCiCorrect = 0
genSRCiTotal = 0


d1p1 = 0
d1p2 = 0
d1po = 0
d2p1 = 0
d2p2 = 0
d2po = 0
dnp1 = 0
dnp2 = 0
dnpo = 0
other_crain = 0

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
                        if len(word2index.keys()) == 81:
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


                        encoder1 = encoder1.to(device=available_device)
                        decoder1 = decoder1.to(device=available_device)

                        encoder1.load_state_dict(torch.load(directory_now + "/" + enc))
                        decoder1.load_state_dict(torch.load(directory_now + "/" + dec))


                        
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

                                correct_words = target_sent.split()#batch[index][1].split()
                                if not two_agreeing_auxes(target_sent):
                                    break

                                total += 1

                                rc_cat = rc_category(input_sent)#batch[index][0])
                                if rc_cat == "ORC":
                                    genORCTotal += 1
                                    this_orc_total += 1
                                elif rc_cat == "SRC_t":
                                    genSRCtTotal += 1
                                    this_srct_total += 1
                                elif rc_cat == "SRC_i":
                                    genSRCiTotal += 1
                                    this_srci_total += 1
                                                            

                                if pred_sent.split()[0] == target_sent.split()[0]:#batch[index][1].split()[0]:
                                    right += 1
                                    if rc_cat == "ORC":
                                        genORCCorrect += 1
                                        this_orc += 1
                                    elif rc_cat == "SRC_t":
                                        genSRCtCorrect += 1
                                        this_srct += 1
                                    elif rc_cat == "SRC_i":
                                        genSRCiCorrect += 1
                                        this_srci += 1
                                                            

                                elif pred_sent.split()[0] in target_sent.split() and pred_sent.split()[0] in auxes:
                                    first_aux += 1
                                elif pred_sent.split()[0] in auxes:
                                    other_aux += 1
                                else:
                                    other_word += 1

                                if pred_sent == target_sent:#batch[index][1]:
                                    full_right += 1
                                if sent_to_pos(pred_sent) == sent_to_pos(target_sent):#batch[index][1]):
                                    full_right_pos += 1

                                crain_class = crain(input_sent, pred_sent)#batch[index][0], this_sent_final)
                                if crain_class == "d1p1":
                                    d1p1 += 1
                                    this_d1p1 += 1
                                elif crain_class == "d1p2":
                                    d1p2 += 1
                                    this_d1p2 += 1
                                elif crain_class == "d1po":
                                    d1po += 1
                                    this_d1po += 1
                                elif crain_class == "d2p1":
                                    d2p1 += 1
                                    this_d2p1 += 1
                                elif crain_class == "d2p2":
                                    d2p2 += 1
                                    this_d2p2 += 1
                                elif crain_class == "d2po":
                                    d2po += 1
                                    this_d2po += 1
                                elif crain_class == "dnp1":
                                    dnp1 += 1
                                    this_dnp1 += 1
                                elif crain_class == "dnp2":
                                    dnp2 += 1
                                    this_dnp2 += 1
                                elif crain_class == "dnpo":
                                    dnpo += 1
                                    this_dnpo += 1
                                else:
                                    other_crain += 1
                                    this_other += 1

                print("Number of sentences with the correct prediction:", right)
                print("Number of sentences fully correct:", full_right)
                print("Total number of sentences", total)
		
                allTotalGen += total
                allGenCorrect += right
                allGenFirstAux += first_aux
                allGenOtherAux += other_aux
                allGenOtherWord += other_word
                allGenFullsent += full_right
                allGenFullsentPos += full_right_pos
                
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

print("Overall gen first word correct aux:", allGenCorrect)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenCorrect * 1.0 / allTotalGen)
print("Gen first word list:")
print(", ".join([str(x) for x in gen_first_word]))
print("Mean:", str(mean(gen_first_word)))
print("Median:", str(median(gen_first_word)))
print("Mean10:", str(mean(gen_first_word[:10])))
print("Median10:", str(median(gen_first_word[:10])))
print(" ")
print("Overall gen first word first aux:", allGenFirstAux)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenFirstAux * 1.0 / allTotalGen)
print("Gen first word list first:")
print(", ".join([str(x) for x in gen_first_word_first_aux]))
print("Mean:", str(mean(gen_first_word_first_aux)))
print("Median:", str(median(gen_first_word_first_aux)))
print("Mean10:", str(mean(gen_first_word_first_aux[:10])))
print("Median10:", str(median(gen_first_word_first_aux[:10])))
print(" ")
print("Overall gen first word other aux:", allGenOtherAux)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenOtherAux * 1.0 / allTotalGen)
print("Gen first word list:")
print(", ".join([str(x) for x in gen_first_word_other_aux]))
print("Mean:", str(mean(gen_first_word_other_aux)))
print("Median:", str(median(gen_first_word_other_aux)))
print("Mean10:", str(mean(gen_first_word_other_aux[:10])))
print("Median10:", str(median(gen_first_word_other_aux[:10])))
print(" ")
print("Overall gen first word other word:", allGenOtherWord)
print("Overall gen total:", allTotalGen)
print("Overall gen accuracy", allGenOtherWord * 1.0 / allTotalGen)
print("Gen first word list:")
print(", ".join([str(x) for x in gen_first_word_other_word]))
print("Mean:", str(mean(gen_first_word_other_word)))
print("Median:", str(median(gen_first_word_other_word)))
print("Mean10:", str(mean(gen_first_word_other_word[:10])))
print("Median10:", str(median(gen_first_word_other_word[:10]))) 
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



print("d1p1", d1p1 * 1.0 / allTotalGen)
print("d1p1 list:")
print(", ".join([str(x) for x in d1p1_lst]))
print("Mean:", str(mean(d1p1_lst)))
print("Median:", str(median(d1p1_lst)))
print("Mean10:", str(mean(d1p1_lst[:10])))
print("Median10:", str(median(d1p1_lst[:10])))
print(" ")
print("d1p2", d1p2 * 1.0 / allTotalGen)
print("d1p2 list:")
print(", ".join([str(x) for x in d1p2_lst]))
print("Mean:", str(mean(d1p2_lst)))
print("Median:", str(median(d1p2_lst)))
print("Mean10:", str(mean(d1p2_lst[:10])))
print("Median10:", str(median(d1p2_lst[:10])))
print(" ")
print("d1po", d1po * 1.0 / allTotalGen)
print("d1po list:")
print(", ".join([str(x) for x in d1po_lst]))
print("Mean:", str(mean(d1po_lst)))
print("Median:", str(median(d1po_lst)))
print("Mean10:", str(mean(d1po_lst[:10])))
print("Median10:", str(median(d1po_lst[:10])))
print(" ")
print("d2p1", d2p1 * 1.0 / allTotalGen)
print("d2p1 list:")
print(", ".join([str(x) for x in d2p1_lst]))
print("Mean:", str(mean(d2p1_lst)))
print("Median:", str(median(d2p1_lst)))
print("Mean10:", str(mean(d2p1_lst[:10])))
print("Median10:", str(median(d2p1_lst[:10])))
print(" ")
print("d2p2", d2p2 * 1.0 / allTotalGen)
print("d2p2 list:")
print(", ".join([str(x) for x in d2p2_lst]))
print("Mean:", str(mean(d2p2_lst)))
print("Median:", str(median(d2p2_lst)))
print("Mean10:", str(mean(d2p2_lst[:10])))
print("Median10:", str(median(d2p2_lst[:10])))
print(" ")
print("d2po", d2po * 1.0 / allTotalGen)
print("d2po list:")
print(", ".join([str(x) for x in d2po_lst]))
print("Mean:", str(mean(d2po_lst)))
print("Median:", str(median(d2po_lst)))
print("Mean10:", str(mean(d2po_lst[:10])))
print("Median10:", str(median(d2po_lst[:10])))
print(" ")
print("dnp1", dnp1 * 1.0 / allTotalGen)
print("dnp1 list:")
print(", ".join([str(x) for x in dnp1_lst]))
print("Mean10:", str(mean(dnp1_lst[:10])))
print("Median10:", str(median(dnp1_lst[:10])))
print(" ")
print("dnp2", dnp2 * 1.0 / allTotalGen)
print("dnp2 list:")
print(", ".join([str(x) for x in dnp2_lst]))
print("Mean:", str(mean(dnp2_lst)))
print("Median:", str(median(dnp2_lst)))
print("Mean10:", str(mean(dnp2_lst[:10])))
print("Median10:", str(median(dnp2_lst[:10])))
print(" ")
print("dnpo", dnpo * 1.0 / allTotalGen)
print("dnpo list:")
print(", ".join([str(x) for x in dnpo_lst]))
print("Mean:", str(mean(dnpo_lst)))
print("Median:", str(median(dnpo_lst)))
print("Mean10:", str(mean(dnpo_lst[:10])))
print("Median10:", str(median(dnpo_lst[:10])))
print(" ")
print("other", other_crain * 1.0 / allTotalGen)
print("other list:")
print(", ".join([str(x) for x in other_lst]))
print("Mean:", str(mean(other_lst)))
print("Median:", str(median(other_lst)))
print("Mean10:", str(mean(other_lst[:10])))
print("Median10:", str(median(other_lst[:10]))) 
print("")
#print("other", other_crain)
#print("d1p2:", d1p2)

print("ORC correct aux:", genORCCorrect * 1.0 / genORCTotal)
print("ORC list:")
print(", ".join([str(x) for x in orc_lst]))
print("Mean:", str(mean(orc_lst)))
print("Median:", str(median(orc_lst)))
print("Mean10:", str(mean(orc_lst[:10])))
print("Median10:", str(median(orc_lst[:10])))
print("")

print("SRC_t correct aux:", genSRCtCorrect * 1.0 / genSRCtTotal)
print("SRC_t list:")
print(", ".join([str(x) for x in srct_lst]))
print("Mean:", str(mean(srct_lst)))
print("Median:", str(median(srct_lst)))
print("Mean10:", str(mean(srct_lst[:10])))
print("Median10:", str(median(srct_lst[:10])))
print("")

print("SRC_i correct aux:", genSRCiCorrect * 1.0 / genSRCiTotal)
print("SRC_i list:")
print(", ".join([str(x) for x in srci_lst]))
print("Mean:", str(mean(srci_lst)))
print("Median:", str(median(srci_lst)))
print("Mean10:", str(mean(srci_lst[:10])))
print("Median10:", str(median(srci_lst[:10])))
print("")

print("ORC correct:", genORCCorrect)
print("ORC total:", genORCTotal)
print("SRC_t correct:", genSRCtCorrect)
print("SRC_t total:", genSRCtTotal)
print("SRC_i correct:", genSRCiCorrect)
print("SRC_i total:", genSRCiTotal)




