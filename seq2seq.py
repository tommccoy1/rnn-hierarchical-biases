
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


from evaluation import *
from models import *
from parsing import *
from training import *


# Start-of-sentence and end-of-sentence tokens
# The standard seq2seq version only has one EOS. This version has 
# 2 EOS--one signalling that the original sentence should be returned,
# the other signalling it should be reversed.
# I use a 1-hot encoding for all tokens.
SOS_token = 0
EOS_tokenA = 1 # For DECL
EOS_tokenB = 2 # For QUEST

prefix = sys.argv[1] 
directory = sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + "_" + sys.argv[6]


if __name__ == "__main__":
        wait_time = random.randint(0, 99)
        time.sleep(wait_time)

        counter = 0
        dir_made = 0

        while not dir_made:
                if not os.path.exists(directory + "_" +  str(counter)):
                        directory = directory + "_" + str(counter)
                        os.mkdir(directory)
                        dir_made = 1

                else:
                        counter += 1

        random_seed = counter
        random.seed(random_seed)

# Reading the training data
trainingFile = prefix + '.train'
devFile = prefix + '.dev'
testFile = prefix + '.test'
genFile = prefix + '.gen'

batch_size = 5
if "TREE" in sys.argv[3]:
	batch_size = 1 # ADDED


use_cuda = torch.cuda.is_available()

if use_cuda:
	available_device = torch.device('cuda')
else:
	available_device = torch.device('cpu')

word2index = {}
index2word = {}

#word2index["SOS"] = 0
#word2index["."] = 1
#word2index["?"] = 2
#index2word[0] = "SOS"
#index2word[1] = "."
#index2word[2] = "?"

if sys.argv[1] == "tense":
	fi = open("index_tense.txt", "r")
else:
	fi = open("index.txt", "r")


for line in fi:
	parts = line.strip().split("\t")
	word2index[parts[0]] = int(parts[1])
	index2word[int(parts[1])] = parts[0]

MAX_LENGTH = 20

def file_to_batches(filename, MAX_LENGTH):
	fi = open(filename, "r")
	pairs = []

	count_sents = 0 # DELETE
	for line in fi:
		parts = line.strip().split("\t")
		s1 = parts[0].strip().lower()
		s2 = parts[1].strip().lower()

		if sys.argv[1] != "tense" and (("TREE" in sys.argv[3] and "NOPRE" not in sys.argv[3]) or "PROC" in sys.argv[3]):
			s1 = preprocess(s1)
			s2 = preprocess(s2)


		#for word in s1.split():
		#	if word not in word2index:
		#		index = len(word2index.keys())
		#		word2index[word] = index
		#		index2word[index] = word

		words1 = [word2index[word] for word in s1.split()]
		words2 = [word2index[word] for word in s2.split()]

		words1 = Variable(torch.LongTensor(words1).view(-1, 1)).to(device=available_device)
		words2 = Variable(torch.LongTensor(words2).view(-1, 1)).to(device=available_device)

		if sys.argv[1] == "tense":
			#fo = open("tense_dir/" + s1.replace(" ", "_"), "w")
			#fo.write("blank")
			pair = [words1, words2, parse_tense(s1), parse_tense(s2)]
		else:
			if "NOPRE" in sys.argv[3]:
				pair = [words1, words2, parse_nopre(s1), parse_nopre(s2)]
			else:
				pair = [words1, words2, pos_to_parse(sent_to_pos_tree(s1)), pos_to_parse(sent_to_pos_tree(s2))]
		pairs.append(pair)

	length_sorted_pairs_dict = {}
	
	for i in range(30):
		length_sorted_pairs_dict[i] = []  

	for pair in pairs:
		length = len(pair[0])
		if length not in length_sorted_pairs_dict:
			length_sorted_pairs_dict[length] = []
		length_sorted_pairs_dict[length].append(pair)

		if length > MAX_LENGTH:
			MAX_LENGTH = length

	length_sorted_pairs_list = []

	for i in range(30):
		possibilities = length_sorted_pairs_dict[i]
		random.shuffle(possibilities)

		this_set = []
		for j in range(len(possibilities)):
			this_set.append(possibilities[j])
			if len(this_set) == batch_size:
				length_sorted_pairs_list.append(this_set)
				this_set = []
				random.shuffle(length_sorted_pairs_list)


	batch_list = []
	for pre_batch in length_sorted_pairs_list:
		tensorA = None
		tensorB = None
		
		for elt in pre_batch:
			if tensorA is None:
				tensorA = elt[0]
				tensorB = elt[1]
			else:
				tensorA = torch.cat((tensorA, elt[0]), 1)
				tensorB = torch.cat((tensorB, elt[1]), 1)
		
		batch_list.append([tensorA, tensorB, elt[2], elt[3]])

	return batch_list, MAX_LENGTH
if __name__ == "__main__":

	train_batches, MAX_LENGTH = file_to_batches(trainingFile, MAX_LENGTH)
	dev_batches, MAX_LENGTH = file_to_batches(devFile, MAX_LENGTH)
	test_batches, MAX_LENGTH = file_to_batches(testFile, MAX_LENGTH)
	gen_batches, MAX_LENGTH = file_to_batches(genFile, MAX_LENGTH)

	print("all batchified")



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


	# Where the actual running of the code happens
	hidden_size = int(sys.argv[6]) # Default = 128

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

	manual_lr = float(sys.argv[5])
	
	torch.manual_seed(random_seed)
	if use_cuda:
		torch.cuda.manual_seed_all(random_seed)	

	print("starting to train")

	if recurrent_unit == "SRN":
		# Default learning rate: 0.001
		trainIters(encoder1, decoder1, 10000000, recurrent_unit, attention, train_batches, dev_batches, index2word, directory, prefix, print_every=1000, learning_rate=manual_lr)
	elif attention == 2:
		# Default learning rate: 0.005
		trainIters(encoder1, decoder1, 10000000, recurrent_unit, attention, train_batches, dev_batches, index2word, directory, prefix, print_every=1000, learning_rate=manual_lr)
	else:
		# Default learning rate: 0.01
		trainIters(encoder1, decoder1, 10000000, recurrent_unit, attention, train_batches, dev_batches, index2word, directory, prefix, print_every=1000, learning_rate=manual_lr)








