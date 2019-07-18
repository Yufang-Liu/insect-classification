#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/06/30 16:21:35

@author: Changzhi Sun
"""
import configargparse 
import random
import os
import time
import sys
import re
import pickle
sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np

from src.model import MLP, CNN
from lib.mydataset import ImgDataset
from train import validation_step

def build_dataset_iter(opt, vocab):
    
    dataset = ImgDataset(opt.data_dir, opt.test_path, vocab)
    print("test data size: %d" % (len(dataset)))

    dataset_iter = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False)
    return dataset, dataset_iter


def build_model(opt, vocab, checkpoint, gpu_id=None):
    print('Building model...')
    if opt.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu_id)
    #  model = MLP([1000, 500, 500], len(vocab))
    model = CNN(len(vocab))

    if checkpoint is not None:
        # end of patch for backward compatibility
        print("Loading model parameters from checkpoint...")
        model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    return model, device

def main(opt):
    print('Loading checkpoint from %s' % opt.model)
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    print('Loading vocab from checkpoint at %s' % opt.model)
    vocab = checkpoint['vocab']
    start_epoch = checkpoint['epoch'] + 1
    val_score = checkpoint['val_score']
    print('Epoch of checkpoint is %d' % checkpoint['epoch'])
    print('Validaton score of checkpoint is %f' % checkpoint['val_score'])

    #  print(vocab)
    print(' * vocab size = %d' % (len(vocab)))
    model, device = build_model(opt, vocab, checkpoint)
    print(model)

    print("Building test data iterator...")
    test_dataset, test_iter = build_dataset_iter(opt, vocab)

    if opt.gpu != -1:
        print('Starting evalation on GPU: %s' % opt.gpu)
    else:
        print('Starting evaluation on CPU, could be very slow')

    print("Evaluating test set...")
    start = time.time()
    metric, total_ins = validation_step(model, test_iter, test_dataset, opt, device)
    elapsed = time.time() - start

    for n in [1, 10, 20, 50, 100]:
        print("P@%-3d: %f" % (n, metric["P@%d" % n]))
    print("MAP  : %f" % (metric["MAP"]))

def _get_parser():
    parser = configargparse.ArgumentParser(description="evaluation.py")
    group = parser.add_argument_group('Model')
    group.add('--model', '-model', required=True,
                help="Path to model .pt file(s). ")

    group = parser.add_argument_group('Data')
    group.add('--data_dir', '-data_dir', required=True,
              help="Directory to the all data")
    group.add('--test_path', '-test_path', required=True,
              help="Path to the test data")

    group = parser.add_argument_group('Efficiency')
    group.add('--batch_size', '-batch_size', type=int, default=64,
              help='Maximum batch size for training')
    group.add('--gpu', '-gpu', type=int, default=-1,
              help="Device to run on")
    return parser

if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    for key, value  in vars(opt).items():
        print("{0} = {1}".format(key, value))
    main(opt)

