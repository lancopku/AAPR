# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:04:44 2018

@author: ypc
"""

import os
import torch
import torch.utils.data as torch_data
import data.utils


class dataset(torch_data.Dataset):

    def __init__(self, text_data, label_data):
        self.text_data = text_data
        self.label_data = label_data
  
    def __getitem__(self, index):
        return [torch.from_numpy(x[index]).type(torch.FloatTensor) for x in self.text_data],\
               torch.from_numpy(self.label_data[index]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.label_data)
       

def get_loader(dataset, batch_size, shuffle, num_workers):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader