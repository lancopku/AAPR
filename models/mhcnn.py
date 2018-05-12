# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:48:24 2018

@author: ypc
"""

import os
import json as js
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lstm import Basic_LSTM
from models.attention import Attention, Conv_attention


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


class Basic_CNN(nn.Module):
    def __init__(self, embed_size, filter_sizes, num_filters,
                 dropout=0.5, l2_reg_lambda=0.0):
        super(Basic_CNN, self).__init__()

        self.convs = nn.ModuleList()
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            each_conv = nn.Sequential(nn.Conv2d(in_channels=1,
                                                out_channels=num_filter,
                                                kernel_size=(filter_size, embed_size)),
                                      nn.ReLU())
            self.convs.append(each_conv)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input:
            x: Variable, tensor, [batch, in_channels, seq_len, embed_size]
        Return:
            Variable, tensor, [batch, sum(num_filters)]
        """

        x_avg_pooled = [F.avg_pool2d(conv(x), conv(x).size()[-2:]).squeeze() for conv in self.convs]
        x_max_pooled = [F.max_pool2d(conv(x), conv(x).size()[-2:]).squeeze() for conv in self.convs]
        x_avg_pooled = torch.cat(x_avg_pooled, dim=1)
        x_max_pooled = torch.cat(x_max_pooled, dim=1)
        return torch.cat((x_avg_pooled, x_max_pooled), dim=1)


class MHCNN(nn.Module):
    def __init__(self, config, share_vocab=True, share_nn=True, predict_way='attention'):
        super(MHCNN, self).__init__()

        self.share_vocab = share_vocab
        self.share_nn = share_nn
        self.predict_way = predict_way
        self.authors_seq_len = config.authors_seq_len
        self.final_dim = sum(config.num_filters2[0])

        if share_vocab:
            self.text_embedding = nn.Embedding(config.vocab_size[0], config.embed_size[0])
            self.authors_embedding = nn.Embedding(config.vocab_size[-1], config.embed_size[-1])
        else:
            self.embeddings = nn.ModuleList()
            for vocab_size, embed_size in zip(config.vocab_size, config.embed_size):
                self.embeddings.append(nn.Embedding(vocab_size, embed_size))

        if share_nn:
            self.cnn1 = Basic_CNN(config.embed_size[0], config.filter_sizes1[0],
                                  config.num_filters1[0], config.cnn_dropout[0],
                                  config.l2_reg_lambda[0])
            self.cnn2 = Basic_CNN(sum(config.num_filters1[0]), config.filter_sizes2[0],
                                  config.num_filters2[0], config.cnn_dropout[0],
                                  config.l2_reg_lambda[0])
        else:
            self.cnns1 = nn.ModuleList()
            for i in range(config.text_numbers - 1):
                cnn = Basic_CNN(config.embed_size[i], config.filter_sizes1[i], config.num_filters1[i],
                                config.cnn_dropout[i], config.l2_reg_lambda[i])
                self.cnns1.append(cnn)
            self.cnns2 = nn.ModuleList()
            for i in range(config.text_numbers):
                cnn = Basic_CNN(sum(config.num_filters1[i]), config.filter_sizes2[i], config.num_filters2[i],
                                config.cnn_dropout[i], config.l2_reg_lambda[i])
                self.cnns2.append(cnn)

        self.authors_coffs = nn.Linear(config.authors_seq_len, 1)

        if self.predict_way == 'linear_combination':
            self.combination_coffs = nn.Parameter(torch.Tensor(1, config.text_numbers + 1))
            self.linear = nn.Linear(self.final_dim, config.num_classes)
        elif self.predict_way == 'single_value':
            self.linear1 = nn.Linear(self.final_dim, 2)
            self.linear2 = nn.Linear(2*(config.text_numbers + 1), config.num_classes)
        elif self.predict_way == 'concat':
            self.mlp = nn.Sequential(nn.Linear((config.text_numbers + 1) * self.final_dim, self.final_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.final_dim, config.num_classes))
        elif self.predict_way == 'attention':
            self.attention = nn.Sequential(Attention(self.final_dim), nn.Linear(self.final_dim, config.num_classes))
        elif self.predict_way == 'lstm':
            self.pre_lstm = Basic_LSTM(self.final_dim,
                                     config.pre_lstm_hidden_size,
                                     config.pre_lstm_num_layers,
                                     config.pre_lstm_dropout,
                                     config.pre_lstm_bi_direc,
                                     config.pre_lstm_return_average, False)
            self.linear = nn.Linear(config.pre_lstm_final_dim, config.num_classes)                                
        else:
            raise ValueError("Please input correct predict_way")

    def forward(self, x):
        """
        Input:
            x: list, the order is [introduction, related work,
                                   methods, conclusion, abstract, title, authors];
               Each element is Variable (maybe cuda);
               The shape of variables except title and authors is [batch, max_doc_len, max_seq_len];
               Different element may has different max_seq_len.
        Return:
            y: Variable
        """

        batch_size = x[0].size(0)
        max_doc_len = []
        max_seq_len = []
        for i in range(len(x) - 2):
            max_doc_len.append(x[i].size(1))
            max_seq_len.append(x[i].size(2))
        # final x_embed is list.
        # each element of x_embed is Variable, [batch, max_seq_len, embed_size]
        x_embed = []
        if self.share_vocab:
            for i in range(len(x) - 2):
                x_embed.append(self.text_embedding(x[i].view(-1, max_seq_len[i])))
            x_embed.append(self.text_embedding(x[-2]))
            x_embed.append(self.authors_embedding(x[-1]))
        else:
            for i in range(len(x) - 2):
                x_embed.append(self.embeddings[i](x[i].view(-1, max_seq_len[i])))
            x_embed.append(self.embeddings[-2](x[-2]))
            x_embed.append(self.embeddings[-1](x[-1]))

        x_conved1 = []
        if self.share_nn:
            for i in range(len(x_embed) - 2):
                each_x_conved = self.cnn1(torch.unsqueeze(x_embed[i], 1)).view(batch_size, max_doc_len[i], -1)
                x_conved1.append(each_x_conved)
        else:
            for i in range(len(x_embed) - 2):
                each_x_conved = self.cnns1[i](torch.unsqueeze(x_embed[i], 1)).view(batch_size, max_doc_len[i], -1)
                x_conved1.append(each_x_conved)

        x_conved2 = []
        if self.share_nn:
            for i in range(len(x_conved1)):
                each_x_conved = self.cnn2(x_conved1[i].unsqueeze(1))
                x_conved2.append(each_x_conved)
            x_conved2.append(self.cnn2(x_embed[-2].unsqueeze(1)))
        else:
            for i in range(len(x_conved1)):
                each_x_conved = self.cnns2[i](x_conved1[i].unsqueeze(1))
                x_conved2.append(each_x_conved)
            x_conved2.append(self.cnns2[-1](x_embed[-2].unsqueeze(1)))
                
        batch_size, authors_seq_len, authors_dim = x_embed[-1].size()
        authors = x_embed[-1].transpose(1, 2).contiguous().view(-1, authors_seq_len)
        authors = self.authors_coffs(authors).squeeze().view(batch_size, -1)
        
        x_conved2.append(authors)
        x_conved = torch.stack(x_conved2, dim=1)
        batch_size, seq_len, final_dim = x_conved.size()

        '''
        with codecs.open('record', 'a', 'utf-8') as f:
            f.writelines(self.attention(x_conved).data.tolist)
        '''

        if self.predict_way == 'linear_combination':
            a, b = self.combination_coffs.size()
            return self.linear(torch.bmm(self.combination_coffs.expand((batch_size, a, b)), x_conved).squeeze())
        elif self.predict_way == 'single_value':
            result = self.linear2(F.relu(self.linear1(x_conved.view(-1, final_dim))).view(batch_size, -1))
            #record_mid('single_value', result.data)
            return result
        elif self.predict_way == 'concat':
            return self.mlp(x_conved.view(batch_size, -1))
        elif self.predict_way == 'attention':
            a = self.attention(x_conved)
            #record_mid('attention', a.data)
            return a
        elif self.predict_way == 'lstm':
            return self.linear(self.pre_lstm(x_conved, []))
        else:
            raise ValueError("Please input correct predict_way")