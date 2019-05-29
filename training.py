
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

# Training the seq2seq network
def train(training_pair_set, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, recurrent_unit, attention=False):
    loss = 0

    training_pair = training_pair_set
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    batch_size = training_pair[0].size()[1]

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_output, encoder_hidden, encoder_outputs = encoder(training_pair_set)

    decoder_hidden = encoder_hidden

    decoder_outputs = decoder(decoder_hidden, encoder_outputs, training_pair_set)

    for di in range(target_variable.size()[0]):
        if di >= len(decoder_outputs):
            break
        loss += criterion(decoder_outputs[di], target_variable[di])

    loss = loss * batch_size
    if not isinstance(loss, int):
        loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length



# Training iterations
def trainIters(encoder, decoder, n_iters, recurrent_unit, attention, train_batches, dev_batches, index2word, directory, prefix, print_every=1000, plot_every=100, learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Training with stochastic gradient descent
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    num_pairs = len(train_batches)

    training_pairs = [random.choice(train_batches)
                          for i in range(100000)]

    criterion = nn.NLLLoss()

    count_since_improved = 0
    best_loss = float('inf')

    print("starting iters")


    # Each weight update
    for iter in range(1, n_iters + 1):
        # The iterations we're looping over for this batch
        training_pair_set = training_pairs[(iter - 1)%len(training_pairs)]# * batch_size:iter * batch_size]

    #    print("in iters")

        loss = train(training_pair_set, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, recurrent_unit, attention)

     #   print("done first iter")
        print_loss_total += loss
        plot_loss_total += loss

        # For saving the weights, if you want to do that
        if (iter % 1000 == 0 and "TREE" not in sys.argv[3]) or (("TREE" in sys.argv[3]) and iter % 5000 == 0): # CHANGE BACK: 10 -> 1000
                dev_set_loss = 1 - score(dev_batches, encoder, decoder, index2word) #dev_loss(dev_batches, encoder, decoder, criterion, recurrent_unit, attention)

                #torch.save(encoder.state_dict(), directory + "/" + prefix + ".encoder." + str(dev_set_loss) + "." + str(iter))
                #torch.save(decoder.state_dict(), directory + "/" + prefix + ".decoder." + str(dev_set_loss) + "." + str(iter))
                dummy_file = open(directory + "/" + prefix + ".loss." + str(dev_set_loss) + ".iter." + str(iter), "w")
                dummy_file.write(" ")

                print("doing loss")

                if dev_set_loss <= best_loss:
                    if "TREE" in sys.argv[3]:
                        if iter % 5000 == 0:
                            torch.save(encoder.state_dict(), directory + "/" + prefix + ".encoder." + "0.0" + "." + "0")
                            torch.save(decoder.state_dict(), directory + "/" + prefix + ".decoder." + "0.0" + "." + "0")
                        print("in dev loss checker", dev_set_loss)

                    else:
                        torch.save(encoder.state_dict(), directory + "/" + prefix + ".encoder." + "0.0" + "." + "0")
                        torch.save(decoder.state_dict(), directory + "/" + prefix + ".decoder." + "0.0" + "." + "0")
                        print("in other dev set loss", dev_set_loss)

                if "TREE" in sys.argv[3]:
                    if iter % 5000 == 0:
                        if dev_set_loss < best_loss:
                            best_loss = dev_set_loss
                            count_since_improved = 0
                        else:
                            count_since_improved += 1
                    print("in dev counter")
                    if count_since_improved >= 3 and iter >= 150000: #500000: #150000: # Changed to make trees train longer
                        break


                else:
                    print("in other dev counter")
                    if dev_set_loss < best_loss:
                        best_loss = dev_set_loss
                        count_since_improved = 0
                    else:
                        count_since_improved += 1

                    if count_since_improved >= 3 and iter >= 30000:
                        break




