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

use_cuda = torch.cuda.is_available()

if use_cuda:
        available_device = torch.device('cuda')
else:
        available_device = torch.device('cpu')

# Class for the encoder RNN
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, recurrent_unit, n_layers=1, max_length=30):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_type = recurrent_unit
        self.max_length = max_length

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(hidden_size, hidden_size)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(hidden_size, hidden_size)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(hidden_size, hidden_size)
        elif recurrent_unit == "MyLSTM":
                self.rnn = MyLSTM(hidden_size, hidden_size)
        elif recurrent_unit == "LSTMSqueeze":
                self.rnn = LSTMSqueeze(hidden_size, hidden_size)
        elif recurrent_unit == "LSTMBob":
                self.rnn = LSTMBob(hidden_size, hidden_size)
        elif recurrent_unit == "ONLSTM":
                self.rnn = ONLSTM(hidden_size, hidden_size)
        elif recurrent_unit == "GRUUnsqueeze":
                self.rnn = GRUUnsqueeze(hidden_size, hidden_size)
        elif recurrent_unit == "GRUUnsqueeze2":
                self.rnn = GRUUnsqueeze2(hidden_size, hidden_size)
        elif recurrent_unit == "GRUBob":
                self.rnn = GRUBob(hidden_size, hidden_size)
        else:
                print("Invalid recurrent unit type")


    # Creates the initial hidden state
    def initHidden(self, recurrent_unit, batch_size):
        if recurrent_unit == "SRN" or recurrent_unit == "GRU" or recurrent_unit == "TREEDEC" or recurrent_unit == "TREEBOTH" or recurrent_unit == "GRUBob" or recurrent_unit == "GRUUnsqueeze" or recurrent_unit == "GRUUnsqueeze2":
                result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        elif recurrent_unit == "LSTM" or recurrent_unit == "MyLSTM" or recurrent_unit == "LSTMSqueeze" or recurrent_unit == "ONLSTM" or recurrent_unit == "LSTMBob":
                result = (Variable(torch.zeros(1, batch_size, self.hidden_size)), Variable(torch.zeros(1, batch_size, self.hidden_size)))
        else:
                print("Invalid recurrent unit type", recurrent_unit)

        if recurrent_unit == "LSTM" or recurrent_unit == "MyLSTM" or recurrent_unit == "LSTMSqueeze" or recurrent_unit == "ONLSTM" or recurrent_unit == "LSTMBob":
                return (result[0].to(device=available_device), result[1].to(device=available_device))
        else:
                return result.to(device=available_device)



    # For succesively generating each new output and hidden layer
    def forward(self, training_pair):

        input_variable = training_pair[0]
        target_variable = training_pair[1]

        batch_size = training_pair[0].size()[1]

        hidden = self.initHidden(self.rnn_type, batch_size)

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        outputs = Variable(torch.zeros(self.max_length, batch_size, self.hidden_size))
        outputs = outputs.to(device=available_device)

        for ei in range(input_length):
        
            output = self.embedding(input_variable[ei]).unsqueeze(0)#.view(1, 1, -1)
            for i in range(self.n_layers):
                output, hidden = self.rnn(output, hidden)
            outputs[ei] = output

        return output, hidden, outputs

# Class for the basic decoder RNN, without attention
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, recurrent_unit, attn=False, n_layers=1, dropout_p=0.1, max_length=30):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attention = attn

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "MyLSTM":
                self.rnn = MyLSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "LSTMSqueeze":
                self.rnn = LSTMSqueeze(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "LSTMBob":
                self.rnn = LSTMBob(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "ONLSTM":
                self.rnn = ONLSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "GRUUnsqueeze":
                self.rnn = GRUUnsqueeze(hidden_size, hidden_size)
        elif recurrent_unit == "GRUUnsqueeze2":
                self.rnn = GRUUnsqueeze2(hidden_size, hidden_size)
        elif recurrent_unit == "GRUBob":
                self.rnn = GRUBob(hidden_size, hidden_size)
        else:
                print("Invalid recurrent unit type")

        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.recurrent_unit = recurrent_unit

        if attn == 1:
                # Attention vector
                self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

                # Context vector made by combining the attentions
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if attn == 2: # for the other type of attention
                self.v = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
                nn.init.uniform(self.v, -1, 1) # maybe need cuda
                self.attn_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    # For successively generating each new output and hidden layer
    def forward_step(self, input, hidden, encoder_outputs, input_variable):
        output = self.embedding(input).unsqueeze(0)#.view(1, 1, -1)
        output = self.dropout(output)

        attn_weights = None

        batch_size = input_variable.size()[1]

        if self.attention == 1:
                if self.recurrent_unit == "LSTM" or self.recurrent_unit == "MyLSTM" or self.recurrent_unit == "LSTMSqueeze" or self.recurrent_unit == "ONLSTM" or self.recurrent_unit == "LSTMBob":
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0][0]), 1)))
                else:
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0]), 1)))

                attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0,1))
                attn_applied = attn_applied.transpose(0,1)

                output = torch.cat((output[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)


        if self.attention == 2: # For the other type of attention
                input_length = input_variable.size()[0] # Check if this is the right index
                u_i = Variable(torch.zeros(len(encoder_outputs), batch_size))

                u_i = u_i.to(device=available_device)


                for i in range(input_length): # can this be done with just matrix operations (i.e. without a for loop)? (probably)

                        if self.recurrent_unit == "LSTM" or self.recurrent_unit == "MyLSTM" or self.recurrent_unit == "LSTMSqueeze" or self.recurrent_unit == "ONLSTM" or self.recurrent_unit == "LSTMBob":
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0][0].unsqueeze(0), output), 2)))
                        else:
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0].unsqueeze(0), output), 2))) # the view(-1) is probably bad
                        u_i_j = torch.bmm(attn_hidden, self.v.unsqueeze(1).unsqueeze(0))
                        u_i[i] = u_i_j[0].view(-1)


                a_i = F.softmax(u_i.transpose(0,1)) # is it correct to be log softmax?
                attn_applied = torch.bmm(a_i.unsqueeze(1), encoder_outputs.transpose(0,1))

                attn_applied = attn_applied.transpose(0,1)

                output = torch.cat((output[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def forward(self, hidden, encoder_outputs, training_set, tf_ratio=0.5):
        input_variable = training_set[0]
        target_variable = training_set[1]

        batch_size = training_set[0].size()[1]

        decoder_input = Variable(torch.LongTensor([0] * batch_size))
        decoder_input = decoder_input.to(device=available_device)

        decoder_hidden = hidden
        
        decoder_outputs = []

        use_tf = True if random.random() < tf_ratio else False

        if use_tf:
            for di in range(target_variable.size()[0]):
                decoder_output, decoder_hidden, decoder_attention = self.forward_step(
                                decoder_input, decoder_hidden, encoder_outputs, input_variable)
                decoder_input = target_variable[di]
                decoder_outputs.append(decoder_output)

        else:
            for di in range(target_variable.size()[0]): #range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.forward_step(
                            decoder_input, decoder_hidden, encoder_outputs, input_variable) 

                topv, topi = decoder_output.data.topk(1)
                decoder_input = Variable(topi.view(-1))
                decoder_input = decoder_input.to(device=available_device)

                decoder_outputs.append(decoder_output)

                if 1 in topi[0] or 2 in topi[0]:
                    break


        return decoder_outputs 


# Note: nn.Linear contains a bias term within it, which 
# is why no bias term appears in the class below
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wi = nn.Linear(hidden_size + input_size, hidden_size)
        self.wf = nn.Linear(hidden_size + input_size, hidden_size)
        self.wg = nn.Linear(hidden_size + input_size, hidden_size)
        self.wo = nn.Linear(hidden_size + input_size, hidden_size)


    def forward(self, input, hidden):
        hx, cx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        i_t = F.sigmoid(self.wi(input_plus_hidden))
        f_t = F.sigmoid(self.wf(input_plus_hidden))
        g_t = F.tanh(self.wg(input_plus_hidden))
        o_t = F.sigmoid(self.wo(input_plus_hidden))

        cx = f_t * cx + i_t * g_t
        hx = o_t * F.tanh(cx)

        return hx, (hx, cx)


# Note: nn.Linear contains a bias term within it, which 
# is why no bias term appears in the class below
class GRUUnsqueeze(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUUnsqueeze, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wr = nn.Linear(hidden_size + input_size, hidden_size)
        self.wz = nn.Linear(hidden_size + input_size, hidden_size)
        self.wx = nn.Linear(input_size, hidden_size)
        self.urh = nn.Linear(hidden_size, hidden_size)


    def forward(self, input, hidden):
        hx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        r_t = F.sigmoid(self.wr(input_plus_hidden))
        z_t = F.sigmoid(self.wz(input_plus_hidden))
        h_tilde = F.tanh(self.wx(input) + self.urh(r_t * hx))
        h_t = 2*(z_t * hx + (1 - z_t) * h_tilde)

        return h_t, h_t


# Note: nn.Linear contains a bias term within it, which 
# is why no bias term appears in the class below
class GRUUnsqueeze2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUUnsqueeze2, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wr = nn.Linear(hidden_size + input_size, hidden_size)
        self.wz = nn.Linear(hidden_size + input_size, hidden_size)
        self.wx = nn.Linear(input_size, hidden_size)
        self.urh = nn.Linear(hidden_size, hidden_size)


    def forward(self, input, hidden):
        hx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        r_t = F.sigmoid(self.wr(input_plus_hidden))
        h_tilde = F.tanh(self.wx(input) + self.urh(r_t * hx))
        h_t = hx + h_tilde

        return h_t, h_t


# Note: nn.Linear contains a bias term within it, which 
# is why no bias term appears in the class below
class GRUBob(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUBob, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wr = nn.Linear(hidden_size + input_size, hidden_size)
        self.wz = nn.Linear(hidden_size + input_size, hidden_size)
        self.wv = nn.Linear(hidden_size + input_size, hidden_size)
        self.wx = nn.Linear(input_size, hidden_size)
        self.urh = nn.Linear(hidden_size, hidden_size)


    def forward(self, input, hidden):
        hx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        r_t = F.sigmoid(self.wr(input_plus_hidden))
        z_t = F.sigmoid(self.wz(input_plus_hidden))
        v_t = F.sigmoid(self.wv(input_plus_hidden))
        h_tilde = F.tanh(self.wx(input) + self.urh(r_t * hx))
        h_t = z_t * hx + v_t * h_tilde

        return h_t, h_t


# Note: nn.Linear contains a bias term within it, which 
# is why no bias term appears in the class below
class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRU, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wr = nn.Linear(hidden_size + input_size, hidden_size)
        self.wz = nn.Linear(hidden_size + input_size, hidden_size)
        self.wx = nn.Linear(input_size, hidden_size)
        self.urh = nn.Linear(hidden_size, hidden_size)


    def forward(self, input, hidden):
        hx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        r_t = F.sigmoid(self.wr(input_plus_hidden))
        z_t = F.sigmoid(self.wz(input_plus_hidden))
        h_tilde = F.tanh(self.wx(input) + self.urh(r_t * hx))
        h_t = z_t * hx + (1 - z_t) * h_tilde

        return h_t, h_t


class CumMax(nn.Module):
        def __init__(self):
                super(CumMax, self).__init__()

        def forward(self, input):
                #print(nn.Softmax(dim=0)(input))
                #print(nn.Softmax(dim=1)(input))
                #print(nn.Softmax(dim=2)(input))
                #print(torch.cumsum(nn.Softmax()(input), 0))
                #print(torch.cumsum(nn.Softmax()(input), 1))
                #print(torch.cumsum(nn.Softmax(dim=2)(input), 2))

                return torch.cumsum(nn.Softmax(dim=2)(input), 2)

class ONLSTM(nn.Module):
        def __init__(self, input_size, hidden_size):
                super(ONLSTM, self).__init__()

                self.hidden_size = hidden_size
                self.input_size = input_size

                self.wi = nn.Linear(hidden_size + input_size, hidden_size)
                self.wf = nn.Linear(hidden_size + input_size, hidden_size)
                self.wg = nn.Linear(hidden_size + input_size, hidden_size)
                self.wo = nn.Linear(hidden_size + input_size, hidden_size)
                self.wftilde = nn.Linear(hidden_size + input_size, hidden_size)
                self.witilde = nn.Linear(hidden_size + input_size, hidden_size)




        def forward(self, input, hidden):
                hx, cx = hidden
                input_plus_hidden = torch.cat((input, hx), 2)

                f_t = F.sigmoid(self.wf(input_plus_hidden))
                i_t = F.sigmoid(self.wi(input_plus_hidden))
                o_t = F.sigmoid(self.wo(input_plus_hidden))
                c_hat_t = F.tanh(self.wg(input_plus_hidden))

                f_tilde_t = CumMax()(self.wftilde(input_plus_hidden))
                i_tilde_t = 1 - CumMax()(self.witilde(input_plus_hidden))

                omega_t = f_tilde_t * i_tilde_t
                f_hat_t = f_t * omega_t + (f_tilde_t - omega_t)
                i_hat_t = i_t * omega_t + (i_tilde_t - omega_t)

                cx = f_hat_t * cx + i_hat_t * c_hat_t
                hx = o_t * F.tanh(cx)

                return hx, (hx, cx)
# Note: nn.Linear contains a bias term within it, which 
# is why no bias term appears in the class below
class LSTMSqueeze(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMSqueeze, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wi = nn.Linear(hidden_size + input_size, hidden_size)
        self.wf = nn.Linear(hidden_size + input_size, hidden_size)
        self.wg = nn.Linear(hidden_size + input_size, hidden_size)
        self.wo = nn.Linear(hidden_size + input_size, hidden_size)


    def forward(self, input, hidden):
        hx, cx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        i_t = F.sigmoid(self.wi(input_plus_hidden))
        f_t = F.sigmoid(self.wf(input_plus_hidden))
        g_t = F.tanh(self.wg(input_plus_hidden))
        o_t = F.sigmoid(self.wo(input_plus_hidden))

        # Sigmoid as a method to squeeze it
        #cx = F.sigmoid(f_t * cx + i_t * g_t)

        # Halving as a method to squeeze it
        cx = (f_t * cx + i_t * g_t)/2
        hx = o_t * F.tanh(cx)

        return hx, (hx, cx)

# Note: nn.Linear contains a bias term within it, which 
# is why no bias term appears in the class below
class LSTMBob(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMBob, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wi = nn.Linear(hidden_size + input_size, hidden_size)
        self.wf = nn.Linear(hidden_size + input_size, hidden_size)
        self.wg = nn.Linear(hidden_size + input_size, hidden_size)
        self.wo = nn.Linear(hidden_size + input_size, hidden_size)


    def forward(self, input, hidden):
        hx, cx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        i_t = F.sigmoid(self.wi(input_plus_hidden))
        f_t = F.sigmoid(self.wf(input_plus_hidden))
        g_t = F.tanh(self.wg(input_plus_hidden))
        o_t = F.sigmoid(self.wo(input_plus_hidden))

        # Sigmoid as a method to squeeze it
        #cx = F.sigmoid(f_t * cx + i_t * g_t)

        sum_fi = f_t + i_t

        # Halving as a method to squeeze it
        cx = (f_t * cx + i_t * g_t)/sum_fi
        hx = o_t * F.tanh(cx)

        return hx, (hx, cx)

class TreeEncoderRNNNew(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(TreeEncoderRNNNew, self).__init__()
        self.hidden_size = hidden_size
        emb_size = hidden_size
        self.emb_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        #self.hidden_squeeze = nn.Linear(2 * hidden_size, hidden_size)
        #self.rnn = nn.GRU(2 * emb_size, hidden_size)


        self.l_rl = nn.Linear(hidden_size, hidden_size)
        self.r_rl = nn.Linear(hidden_size, hidden_size)

        self.l_rr = nn.Linear(hidden_size, hidden_size)
        self.r_rr = nn.Linear(hidden_size, hidden_size)

        self.l_zl = nn.Linear(hidden_size, hidden_size)
        self.r_zl = nn.Linear(hidden_size, hidden_size)


        self.l_zr = nn.Linear(hidden_size, hidden_size)
        self.r_zr = nn.Linear(hidden_size, hidden_size)

        self.l_z = nn.Linear(hidden_size, hidden_size)
        self.r_z = nn.Linear(hidden_size, hidden_size)

        self.l = nn.Linear(hidden_size, hidden_size)
        self.r = nn.Linear(hidden_size, hidden_size)




    def forward(self, training_set): #input_seq, tree):
        input_seq = training_set[0]
        tree = training_set[2]
        embedded_seq = []

        for elt in input_seq:
            embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt])).to(device=available_device)).unsqueeze(0))
            #embedded_seq.append((self.embedding(Variable(torch.LongTensor([elt])).to(device=available_device)).unsqueeze(0), Variable(torch.zeros(1,1,self.hidden_size).to(device=available_device))))

        current_level = embedded_seq
        for level in tree:
            next_level = []

            for node in level:

                if len(node) == 1:
                    #print("chinchilla", current_level, node[0])
                    next_level.append(current_level[node[0]])
                    continue
                left = node[0]
                right = node[1]


                r_l = nn.Sigmoid()(self.l_rl(current_level[left]) + self.r_rl(current_level[right]))
                r_r = nn.Sigmoid()(self.l_rr(current_level[left]) + self.r_rr(current_level[right]))
                z_l = nn.Sigmoid()(self.l_zl(current_level[left]) + self.r_zl(current_level[right]))
                z_r = nn.Sigmoid()(self.l_zr(current_level[left]) + self.r_zr(current_level[right]))
                z = nn.Sigmoid()(self.l_z(current_level[left]) + self.r_z(current_level[right]))
                h_tilde = nn.Tanh()(self.l(r_l * current_level[left]) + self.r(r_r * current_level[right]))
                hidden = z_l * current_level[left] + z_r * current_level[right] + z * h_tilde

                next_level.append(hidden)

            current_level = next_level

        return current_level[0], current_level[0], current_level[0]



class TreeEncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(TreeEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        emb_size = hidden_size
        self.emb_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.hidden_squeeze = nn.Linear(2 * hidden_size, hidden_size)
        self.rnn = nn.GRU(2 * emb_size, hidden_size)


        self.w_z_l = nn.Linear(emb_size, hidden_size)
        self.w_z_r = nn.Linear(emb_size, hidden_size)

        self.u_z_l = nn.Linear(hidden_size, hidden_size)
        self.u_z_r = nn.Linear(hidden_size, hidden_size)


        self.w_r_l = nn.Linear(emb_size, hidden_size)
        self.w_r_r = nn.Linear(emb_size, hidden_size)

        self.u_r_l = nn.Linear(hidden_size, hidden_size)
        self.u_r_r = nn.Linear(hidden_size, hidden_size)


        self.w_h_l = nn.Linear(emb_size, hidden_size)
        self.w_h_r = nn.Linear(emb_size, hidden_size)

        self.u_h_l = nn.Linear(hidden_size, hidden_size)
        self.u_h_r = nn.Linear(hidden_size, hidden_size)



    def forward(self, training_set): #input_seq, tree):
        input_seq = training_set[0]
        tree = training_set[2]
        embedded_seq = []

        for elt in input_seq:
            embedded_seq.append((self.embedding(Variable(torch.LongTensor([elt])).to(device=available_device)).unsqueeze(0), Variable(torch.zeros(1,1,self.hidden_size).to(device=available_device))))



#            if use_cuda:
#                embedded_seq.append((self.embedding(Variable(torch.LongTensor([elt])).cuda()).unsqueeze(0), Variable(torch.zeros(1,1,self.hidden_size).cuda())))
#            else:
#                embedded_seq.append((self.embedding(Variable(torch.LongTensor([elt]))).unsqueeze(0), Variable(torch.zeros(1,1,self.hidden_size))))

        current_level = embedded_seq
        for level in tree:
            next_level = []

            for node in level:

                if len(node) == 1:
                    #print("chinchilla", current_level, node[0])
                    next_level.append(current_level[node[0]])
                    continue
                left = node[0]
                right = node[1]

                r_t = nn.Sigmoid()(self.u_r_l(current_level[left][1]) + self.w_r_l(current_level[left][0]) + self.u_r_r(current_level[right][1]) + self.w_r_r(current_level[right][0]))
                z_t = nn.Sigmoid()(self.u_z_l(current_level[left][1]) + self.w_z_l(current_level[left][0]) + self.u_z_r(current_level[right][1]) + self.w_z_r(current_level[right][0]))
                h_tilde_t = nn.Sigmoid()(self.u_h_l(current_level[left][1]) + self.w_h_l(current_level[left][0]) + self.u_h_r(current_level[right][1]) + self.w_h_r(current_level[right][0]))
                hidden = z_t * (current_level[left][1] + current_level[right][1]) + (1 - z_t) * h_tilde_t

                next_level.append((Variable(torch.zeros(1,1,self.emb_size).to(device=available_device)), hidden))

            current_level = next_level

        return current_level[0][1], current_level[0][1], current_level[0][1]

class TreeDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(TreeDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_out = nn.Linear(hidden_size, vocab_size)
        self.rnn_l = nn.GRU(hidden_size, hidden_size)
        self.rnn_r = nn.GRU(hidden_size, hidden_size)
        self.left_child = nn.Linear(hidden_size, hidden_size)
        self.right_child = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs, training_set, tf_ratio=0.5): #(self, encoding, tree):
        encoding = hidden
        tree = training_set[3]

        tree_to_use = tree[::-1][1:]

        current_layer = [encoding]

        for layer in tree_to_use:
            next_layer = []
            for index, node in enumerate(layer):
                if len(node) == 1:
                    next_layer.append(current_layer[index])
                else:

                    output, left = self.rnn_l(Variable(torch.zeros(1,1,self.hidden_size)).to(device=available_device), current_layer[index])
                    output, right = self.rnn_r(Variable(torch.zeros(1,1,self.hidden_size)).to(device=available_device), current_layer[index])

                    next_layer.append(left)
                    next_layer.append(right)
            current_layer = next_layer

        words_out = []
        for elt in current_layer:
            words_out.append(nn.LogSoftmax()(self.word_out(elt).view(-1).unsqueeze(0)))

        #print(words_out)
        return words_out



























