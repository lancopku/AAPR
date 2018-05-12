# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 20:18:07 2018

@author: ypc
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Basic_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5,
                 bi_direc=False, return_average=True, need_pack=False):
        super(Basic_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.return_average = return_average
        self.need_pack = need_pack

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bi_direc)

    def forward(self, x, x_len):
        """
        Input:
            x: Variable, tensor, [batch, seq_len, input_size]
            x_len: Tensor, [batch]
        Return:
            Variable, tensor, [batch, num_directions * hidden_size]
        """

        batch_size = x.size(0)
        if self.need_pack:
            x_len_sorted, idx_sorted = torch.sort(x_len, descending=True)
            _, idx_unsort = torch.sort(idx_sorted)
            x_sorted = torch.index_select(x, dim=0, index=idx_sorted)
            lengths = list(np.array(x_len_sorted.data.tolist()))
            # pack:==================================================================
            x_sorted_p = pack(input=x_sorted, lengths=lengths , batch_first=True)

            # process using LSTM: ===================================================
            # out_packed: [batch, Seq_length = time_step, hidden_size]
            # h_t, c_t:   [num_layers * num_directions, batch, hidden_size]
            out_packed, (h_t, c_t) = self.lstm(x_sorted_p, None)
            out_pad, _ = unpack(out_packed, True)
            # unsort h_t and out_pad: ===============================================
            out_pad = torch.index_select(out_pad, dim=0, index=idx_unsort)
            h_t = torch.index_select(h_t, dim=1, index=idx_unsort)
            out_total = torch.sum(out_pad, dim=1).cuda()
            x_len_expand = x_len.unsqueeze(1).expand_as(out_total)
            out_average = torch.div(out_total.type(torch.FloatTensor).cuda(), x_len_expand.type(torch.FloatTensor).cuda())
            h_t = h_t.transpose(0, 1).contiguous().view(batch_size, -1)
        
        else:
            out, (h_t, c_t) = self.lstm(x, None)
            out_average = torch.mean(out, dim=1).cuda()
            h_t = h_t.transpose(0, 1).contiguous().view(batch_size, -1)

        if self.return_average:
            return out_average
        else:
            return h_t