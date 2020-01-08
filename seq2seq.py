
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


# Create a directory where the outputs will be saved
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

# Implement the random seed
if args.seed is None:
    random_seed = counter

random.seed(random_seed)

# Reading the training data
trainingFile = 'data/' + prefix + '.train'
devFile = 'data/' + prefix + '.dev'
testFile = 'data/' + prefix + '.test'
genFile = 'data/' + prefix + '.gen'

# Manually defining the batch size at 5
# Except that, if the model is tree-based at all, the
# batch size must be 1 because you can't easily batch with 
# the Tree-RNNs we are using.
# In those cases, the batching will instead be handled with
# a for loop
batch_size = 5
if args.encoder == "Tree" or args.decoder == "Tree":
	batch_size = 1 

# Determine whether cuda is available
use_cuda = torch.cuda.is_available()

if use_cuda:
	available_device = torch.device('cuda')
else:
	available_device = torch.device('cpu')

# Create dictionaries for converting words to numerical
# indices and vice versa
word2index = {}
index2word = {}

fi = open("index.txt", "r")

for line in fi:
	parts = line.strip().split("\t")
	word2index[parts[0]] = int(parts[1])
	index2word[int(parts[1])] = parts[0]

# Function for preprocessing files into batches
# that can be inputted into our models
MAX_LENGTH = 20
def file_to_batches(filename, MAX_LENGTH, batch_size=5):
	fi = open(filename, "r")
	pairs = []

        # Convert words into indices, and create a parse for each sentence
        # Thus, each training "pair" is really a 4-tuple containing sentence1,
        # sentence2, the parse for sentence1, and the parse for sentence2
	for line in fi:
		parts = line.strip().split("\t")
		s1 = parts[0].strip().lower()
		s2 = parts[1].strip().lower()

		words1 = [word2index[word] for word in s1.split()]
		words2 = [word2index[word] for word in s2.split()]

		words1 = Variable(torch.LongTensor(words1).view(-1, 1)).to(device=available_device)
		words2 = Variable(torch.LongTensor(words2).view(-1, 1)).to(device=available_device)

		if sys.argv[1] == "tense":
			pair = [words1, words2, parse_tense(s1), parse_tense(s2)]
		else:
			pair = [words1, words2, parse_question(s1), parse_question(s2)]
		pairs.append(pair)

        # Now sort these sentence pairs by length, as each batch must
        # have the same length within the batch (you could use padding to
        # avoid this issue, but we didn't do that)
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

        # Convert each batch from a list into a single tensor
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


# Where the training actually happens
if __name__ == "__main__":
        # Convert the input files into batches
        train_batches, MAX_LENGTH = file_to_batches(trainingFile, MAX_LENGTH, batch_size=batch_size)
        dev_batches, MAX_LENGTH = file_to_batches(devFile, MAX_LENGTH, batch_size=batch_size)
        test_batches, MAX_LENGTH = file_to_batches(testFile, MAX_LENGTH, batch_size=batch_size)
        gen_batches, MAX_LENGTH = file_to_batches(genFile, MAX_LENGTH, batch_size=batch_size)

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

        # Give torch a random seed
        torch.manual_seed(random_seed)
        if use_cuda:
            torch.cuda.manual_seed_all(random_seed)	

        # Train the model
        trainIters(encoder, decoder, 10000000, args.encoder, args.decoder, args.attention, train_batches, dev_batches, index2word, directory, prefix, print_every=1000, learning_rate=args.lr)



