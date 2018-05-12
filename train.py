# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:49:40 2018

@author: ypc
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

from data.dataloader import get_loader
import data.utils as utils
from models.mhcnn import MHCNN
from optims import Optim

import os
import codecs
import json as js
import argparse
import time
import collections


# config
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-config', default='config_hcnn_attent_7.json', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-model', default='CNN', type=str,
                    help="Model selection")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-pretrain', default=False, action='store_true',
                    help="load pretrain embedding")
parser.add_argument('-notrain', default=False, action='store_true',
                    help="train or not")
parser.add_argument('-log', default='', type=str,
                    help="log directory")
opt = parser.parse_args()

# config and seed
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# checkpoint
if opt.restore:
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)


# ======================================================================================================================
"""load data and prepare dataloader"""

print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)  
print('loading time cost: %.3f' % (time.time()-start_time))
trainset, testset = datas['train'], datas['val']
trainloader = get_loader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)
valloader = get_loader(valset, batch_size=config.batch_size, shuffle=False, num_workers=2)
print('dataloader prepared')


# ======================================================================================================================
"""prepare pretrained data and model"""

if opt.pretrain:
    pretrain_embed = torch.load(config.emb_file)
else:
    pretrain_embed = None
# built model
print('building model...\n')
model = MHCNN(config, config.share_vocab, config.share_nn, config.predict_way)

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:  
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

# updates
if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

# ======================================================================================================================
"""log config"""

if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'model_config.txt')  
logging_csv = utils.logging_csv(log_path + 'model_record.csv')
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n") 
logging('total number of parameters: %d\n\n' % param_count)

# ======================================================================================================================
"""train"""

scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))
loss_function = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    global e, loss, updates, total_loss, start_time, report_total
    for e in range(1, epoch + 1):
        for x_list, y in trainloader:
            bx = [Variable(x.type(torch.LongTensor)) for x in x_list]
            by = Variable(y.type(torch.FloatTensor))
            if use_cuda:
                bx = [x.cuda() for x in bx]
                by = by.cuda()
            model.zero_grad()
            y_pre = model(bx)
            loss = loss_function(y_pre, torch.max(by, 1)[1])
            loss.backward()
            optim.step()
            updates += 1

            if updates % config.eval_interval == 0:
                print('evaluating after %d updates...\r' % updates)
                score = eval()
                for metric in config.metric:
                    scores[metric].append(score[metric])
                    if score[metric] >= max(scores[metric]):  
                        save_model(log_path + 'best_' + metric + '_checkpoint.pt')
                model.train()

            if updates % config.save_interval == 0:  
                save_model(log_path+'checkpoint.pt')

# ======================================================================================================================
"""eval"""

def eval():
    model.eval()
    y_true, y_pred = [], []
    for x_list, y in valloader:
        bx, by = [Variable(x).type(torch.LongTensor) for x in x_list], Variable(y)
        if use_cuda:
            bx, by = [x.cuda() for x in bx], by.cuda()
        y_pre = model(bx)
        y_label = torch.max(y_pre, 1)[1].data
        y_true.extend(torch.max(y, 1)[1].tolist())
        y_pred.extend(y_label.tolist())

    score = {}
    result = utils.eval_metrics(y_pred, y_true)
    logging_csv([e, updates, loss.data[0], result['accuracy'], result['f1'], result['precision'], result['recall']])
    print('Epoch: %d | Updates: %d | Train loss: %.4f | Accuracy: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f'
          % (e, updates, loss.data[0], result['accuracy'], result['f1'], result['precision'], result['recall']))
    score['accuracy'] = result['accuracy']
    score['f1'] = result['f1']

    return score


def save_model(path):

    global updates
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def main():
    if not opt.notrain:
        train(config.epoch)
    else:
        eval()
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()