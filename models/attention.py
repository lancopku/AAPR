import numpy as np
import os
import json as js
import codecs
import torch
import torch.nn as nn
import torch.nn.functional as F


def record_mid(file, tensor):
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write(js.dumps([]))
    data = tensor.tolist()
    with open(file, 'r') as f:
        result = js.load(f)
    for line in data:
        result.append(line)
    with open(file, 'w') as f:
        f.write(js.dumps(result))


def batch_matmul(seq, weight, nonlinearity='tanh'):
    """
    Inputs:
        seq: tensor, [batch, seq_len, input_size]
        weight: tensor, [input_size, 1]
    Returns:
        tensor, [batch, seq_len]
    """

    seq = seq.transpose(0, 1)
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if nonlinearity == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze().transpose(0, 1)


def batch_matmul1(seq, weight, nonlinearity='tanh'):
    """
    Inputs:
        seq: tensor, [batch, seq_len, input_size]
        weight: tensor, [input_size, 1]
    Returns:
        tensor, [batch, seq_len]
    """

    seq = seq.transpose(0, 1)
    s = None
    for i in range(seq.size(0)):
        _s = torch.sum(seq[i]*weight, dim=1)
        if nonlinearity == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze().transpose(0, 1)


def attention_mul(rnn_outputs, att_weights):
    """
    Inputs:
        rnn_outputs: tensor, [batch, seq_len, input_size]
        att_weights: tensor, [batch, seq_len]
    Return:
        [batch, input_size]
    """

    rnn_outputs = rnn_outputs.transpose(0, 1)
    att_weights = att_weights.transpose(0, 1)
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0)


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()

        self.input_size = input_size
        self.linear_project = nn.Linear(input_size, input_size)
        self.representation = nn.Parameter(torch.Tensor(input_size, 1))
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Input:
            x: Variable, tensor, [batch, seq_len, input_size]
        Return:
            [batch, input_size]
        """

        batch_size, seq_len, input_size = x.size()
        # u: [batch, seq_len, input_size]
        u = self.linear_project(x.contiguous().view(-1, input_size)).contiguous().view(batch_size, seq_len, -1)
        atten_weights = self.softmax(batch_matmul(u, self.representation))
        s = attention_mul(x, atten_weights)
        return s


class Conv_attention(nn.Module):
    def __init__(self, input_size):
        super(Conv_attention, self).__init__()
        
        self.input_size = input_size
        self.linear_project = nn.Linear(input_size, input_size)
        self.softmax = nn.Softmax()

    def forward(self, x, y):
        """
        Input:
            x: Variable, tensor, [batch, seq_len, input_size]
        Return:
            [batch, input_size]
        """

        batch_size, seq_len, input_size = x.size()
        # u: [batch, seq_len, input_size]
        u = self.linear_project(x.view(-1, input_size)).view(batch_size, seq_len, -1)
        atten_weights = self.softmax(batch_matmul1(u, y))
        s = attention_mul(x, atten_weights)
        return s