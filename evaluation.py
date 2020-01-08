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


# Functions for evaluation


# Computing loss on a set of examples
def loss(example_set, encoder, decoder, criterion, recurrent_unit, attention=False, max_length=30):
    loss = 0

    for training_pair in example_set:

        input_variable = training_pair[0]
        target_variable = training_pair[1]
        target_length = target_variable.size()[0]

        encoder_output, encoder_hidden, encoder_outputs = encoder(training_pair)

        decoder_hidden = encoder_hidden

        # No teacher forcing during evaluation
        decoder_outputs = decoder(decoder_hidden, encoder_outputs, training_pair, attn=attention, tf_ratio=0.0)

        for di in range(target_length):
            if len(decoder_outputs) < di:
                break
            loss += criterion(decoder_outputs[di], target_variable[di])/target_length

    loss = loss * batch_size / len(dev_set)

    return loss.data[0]

# Get the model's full-sentence accuracy on a set of examples
def score(example_set, encoder1, decoder1, index2word):
    right = 0
    total = 0

    for unproc_batch in example_set:
        batch = unproc_batch 
        batch_size = batch[0].size()[1]
        elt = batch
        pred_words = evaluate(encoder1, decoder1, elt)

        all_sents = logits_to_sentence(pred_words, index2word)
        correct_sents = logits_to_sentence(batch[1], index2word)

        for sents in zip(all_sents, correct_sents):
            if sents[0] == sents[1]:
                right += 1
            total += 1

    return right * 1.0 / total

# Convert logits to a sentence
def logits_to_sentence(pred_words, index2word, end_at_punc=True):
    batch_size = pred_words.size()[1]
    all_sents = []
    for index in range(batch_size):
        this_sent = []
        for output_word in pred_words: 
            this_sent.append(index2word[output_word[index].item()])
        if end_at_punc:
            if "." in this_sent:
                this_sent = this_sent[:this_sent.index(".") + 1]
            if "?" in this_sent:
                this_sent = this_sent[:this_sent.index("?") + 1]
        this_sent_final = " ".join(this_sent)

        all_sents.append(this_sent_final)

    return all_sents

# Given a batch as input, get the decoder's outputs (as argmax indices)
MAX_EXAMPLE = 10000
def evaluate(encoder, decoder, batch, max_length=30):
    encoder_output, encoder_hidden, encoder_outputs = encoder(batch)

    decoder_hidden = encoder_hidden

    decoder_outputs = decoder(decoder_hidden, encoder_outputs, batch, tf_ratio=0.0)


    output_indices = []
    for logit in decoder_outputs:
        topv, topi = logit.data.topk(1)

        output_indices.append(torch.stack([elt[0] for elt in topi]))

        if 1 in topi or 2 in topi:
            break

    return torch.stack(output_indices)


