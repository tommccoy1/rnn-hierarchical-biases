
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

use_cuda = torch.cuda.is_available()

if use_cuda:
    available_device = torch.device('cuda')
else:
    available_device = torch.device('cpu')

# Train on a single batch, returning the average loss for
# that batch
def train(training_pair, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, attention=False):
    loss = 0

    # Determine the size of the input
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    batch_size = training_pair[0].size()[1]

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Pass the input through the model
    encoder_output, encoder_hidden, encoder_outputs = encoder(training_pair)

    decoder_hidden = encoder_hidden

    decoder_outputs = decoder(decoder_hidden, encoder_outputs, training_pair)

    # Determine the loss
    for di in range(target_variable.size()[0]):
        if di >= len(decoder_outputs):
            break
        loss += criterion(decoder_outputs[di], target_variable[di])

    # Backpropagate the loss
    loss = loss * batch_size
    if not isinstance(loss, int):
        loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data.item() / target_length



# Do many iterations of training
def trainIters(encoder, decoder, n_iters, enc_recurrent_unit, dec_recurrent_unit, attention, training_pairs, dev_batches, index2word, directory, prefix, print_every=1000, learning_rate=0.01, patience=3):
    print_loss_total = 0  

    # Training with stochastic gradient descent
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    random.shuffle(training_pairs)

    criterion = nn.NLLLoss()

    count_since_improved = 0
    best_loss = float('inf')

    # Each iteration is one weight update
    for iter in range(1, n_iters + 1):
        # The current batch
        training_pair_set = training_pairs[(iter - 1)%len(training_pairs)]

        loss = train(training_pair_set, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, attention)

        print_loss_total += loss

        # Evaluate and save every 1000 batches for a sequential model or every 5000 batches
        # for a tree-based model
        if (iter % 1000 == 0 and enc_recurrent_unit != "Tree" and dec_recurrent_unit != "Tree") or (iter % 5000 == 0 and (enc_recurrent_unit == "Tree" or dec_recurrent_unit == "Tree")):
                # Compute the error on the dev set
                dev_set_loss = 1 - score(dev_batches, encoder, decoder, index2word) 

                # Create a blank file whose name shows the current dev set error and the current iteration number
                dummy_file = open(directory + "/" + prefix + ".loss." + str(dev_set_loss) + ".iter." + str(iter), "w")
                dummy_file.write(" ")

                # Determine whether to save the model weights
                # (which is done whenever a new minimum loss is reached)
                if dev_set_loss <= best_loss:
                    torch.save(encoder.state_dict(), directory + "/" + prefix + ".encoder." + "0.0" + "." + "0")
                    torch.save(decoder.state_dict(), directory + "/" + prefix + ".decoder." + "0.0" + "." + "0")
                    print("Dev loss:", dev_set_loss)

                # See if we should early stop
                if enc_recurrent_unit == "Tree" or dec_recurrent_unit == "Tree":
                    if dev_set_loss < best_loss:
                        best_loss = dev_set_loss
                        count_since_improved = 0
                    else:
                        count_since_improved += 1
                    
                    if count_since_improved >= patience and iter >= 150000: # Changed to make trees train longer
                        break


                else:
                    if dev_set_loss < best_loss:
                        best_loss = dev_set_loss
                        count_since_improved = 0
                    else:
                        count_since_improved += 1

                    if count_since_improved >= patience and iter >= 30000:
                        break




