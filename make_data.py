# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:04:44 2018

@author: ypc
"""

import codecs
import json as js
import numpy as np

from data import dict
from data.dataloader import dataset

import torch


def make_text_data1(text_file, text_dict, doc_length, text_length, sep=' '):
    result = []
    with codecs.open(text_file, 'r', 'utf-8') as f:
        data = js.load(f)
    for line in data:
        temp = np.zeros((doc_length, text_length))
        for i in range(len(line)):
            if i < doc_length:
                words = line[i].strip().split(sep)
                for j in range(len(words)):
                    if j < text_length:
                        temp[i, j] = text_dict.lookup(words[j].lower(), 1)
        result.append(temp)
    return result


def make_text_data2(text_file, text_dict, text_length, sep=' '):
    result = []
    with codecs.open(text_file, 'r', 'utf-8') as f:
        data = js.load(f)
    for line in data:
        temp = np.zeros(text_length)
        words = line.strip().split(sep)
        for i in range(len(words)):
            if i < text_length:
                temp[i] = text_dict.lookup(words[i].lower(), 1)
        result.append(temp)
    return result


def make_label_data(label_file, label_dict):
    result = []
    length = len(label_dict)
    with codecs.open(label_file, 'r', 'utf-8') as f:
        data = js.load(f)
    for line in data:
        temp = np.zeros(length)
        temp[label_dict.get(str(line))] = 1
        result.append(temp)
    return result


def make_data(text, title, authors, label):

    text_data = []
    for each_x in text:
        text_data.append(make_text_data1(each_x['text_file'], each_x['text_dict'], \
            each_x['doc_len'], each_x['text_len']))
    text_data.append(make_text_data2(title['text_file'], title['text_dict'], title['text_len']))
    text_data.append(make_text_data2(authors['text_file'], authors['text_dict'], authors['text_len'], sep=','))
    label_data = make_label_data(label['label_file'], label['label_dict'])

    return dataset(text_data, label_data)


def main():
    text_dict = dict.Dict('./data/data/text_dict')
    authors_dict = dict.Dict('./data/data/authors_dict')
    with codecs.open('./data/data/label_dict.json', 'r', 'utf-8') as f:
        label_dict = js.load(f)
    
    intro = {'text_file': './data/data/intro_train', 'text_dict': text_dict, 'doc_len': 100, 'text_len': 25}
    related = {'text_file': './data/data/related_train', 'text_dict': text_dict, 'doc_len': 500, 'text_len': 25}
    methods = {'text_file': './data/data/methods_train', 'text_dict': text_dict, 'doc_len': 600, 'text_len': 25}
    conclu = {'text_file': './data/data/conclusion_train', 'text_dict': text_dict, 'doc_len': 150, 'text_len': 25}
    abstract = {'text_file': './data/data/abstract_train', 'text_dict': text_dict, 'doc_len': 10, 'text_len': 25}
    title = {'text_file': './data/data/title_train', 'text_dict': text_dict, 'text_len': 20}
    authors = {'text_file': './data/data/authors_train', 'text_dict': authors_dict, 'text_len': 7}
    label = {'label_file': './data/data/label_train', 'label_dict': label_dict}
    text = [intro, related, methods, conclu, abstract]
    
    intro_val = {'text_file': './data/data/intro_val', 'text_dict': text_dict, 'doc_len': 100, 'text_len': 25}
    related_val = {'text_file': './data/data/related_val', 'text_dict': text_dict, 'doc_len': 500, 'text_len': 25}
    methods_val = {'text_file': './data/data/methods_val', 'text_dict': text_dict, 'doc_len': 600, 'text_len': 25}
    conclu_val = {'text_file': './data/data/conclusion_val', 'text_dict': text_dict, 'doc_len': 150, 'text_len': 25}
    abstract_val = {'text_file': './data/data/abstract_val', 'text_dict': text_dict, 'doc_len': 10, 'text_len': 25}
    title_val = {'text_file': './data/data/title_val', 'text_dict': text_dict, 'text_len': 20}
    authors_val = {'text_file': './data/data/authors_val', 'text_dict': authors_dict, 'text_len': 7}
    label_val = {'label_file': './data/data/label_val', 'label_dict': label_dict}
    text_val = [intro_val, related_val, methods_val, conclu_val, abstract_val]
     
    intro_test = {'text_file': './data/data/intro_test', 'text_dict': text_dict, 'doc_len': 100, 'text_len': 25}
    related_test = {'text_file': './data/data/related_test', 'text_dict': text_dict, 'doc_len': 500, 'text_len': 25}
    methods_test = {'text_file': './data/data/methods_test', 'text_dict': text_dict, 'doc_len': 600, 'text_len': 25}
    conclu_test = {'text_file': './data/data/conclusion_test', 'text_dict': text_dict, 'doc_len': 150, 'text_len': 25}
    abstract_test = {'text_file': './data/data/abstract_test', 'text_dict': text_dict, 'doc_len': 10, 'text_len': 25}
    title_test = {'text_file': './data/data/title_test', 'text_dict': text_dict, 'text_len': 20}
    authors_test = {'text_file': './data/data/authors_test', 'text_dict': authors_dict, 'text_len': 7}
    label_test = {'label_file': './data/data/label_test', 'label_dict': label_dict}
    text_test = [intro_test, related_test, methods_test, conclu_test, abstract_test]

    train = make_data(text, title, authors, label)
    val = make_data(text_val, title_val, authors_val, label_val)
    test = make_data(text_test, title_test, authors_test, label_test)
    save_data = {'train': train, 'val': val, 'test': test}
    torch.save(save_data, './data/data/save_data')


if __name__ == '__main__':
    main()



