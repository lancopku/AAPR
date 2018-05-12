# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 00:02:11 2018

@author: ypc
"""


from sklearn import metrics
import time
import csv
import json as js


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(js.load(open(path, 'r')))


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)


def logging(file):
    def write_log(s):
        print(s, end='')
        with open(file, 'a') as f:
            f.write(s)
    return write_log


def logging_csv(file):
    def write_csv(s):
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(s)
    return write_csv


def eval_metrics(y_pred, y_true):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')

    return {'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall}
