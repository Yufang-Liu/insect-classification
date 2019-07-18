#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/06/26 17:05:38

@author: Changzhi Sun
"""
import configargparse 
import glob
import sys
import os
import pickle
sys.path.append("..")
import torch
from torchvision import transforms

from lib.mydataset import ImgDataset

def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid']:
        path = pattern.format(t)
        if glob.glob(path):
            print("Please backup existing pt files: %s, "
                  "to avoid overwriting them!\n" % path)
            sys.exit(1)

def build_save_dataset(corpus_type, opt, vocab):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        path = opt.train_path
    elif corpus_type == "valid":
        path = opt.valid_path
    print("Reading file: %s" % (path))
    dataset = ImgDataset(opt.data_dir, path, vocab)

    data_path = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    print(" * saving %s data to %s"
            % (corpus_type, data_path))
    torch.save(dataset, data_path)
    return dataset


def main(opt):
    check_existing_pt_files(opt)

    print("Building & saving vocabulary...")
    vocab = build_save_vocab(opt)
    print("Vocabulary size : %d" % len(vocab))

    print("Building & saving training data...")
    train_data = build_save_dataset('train', opt, vocab)
    print("Training data size : %d" % len(train_data))

    print("Building & saving validation data...")
    valid_data = build_save_dataset('valid', opt, vocab)
    print("Validation data size : %d" % len(valid_data))

def build_save_vocab(opt):
    vocab = {}
    for path in [opt.train_path, opt.valid_path]:
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                assert len(line.split('\t')) == 2
                _, label_str = line.strip().split('\t')
                if label_str not in vocab:
                    vocab[label_str] = len(vocab)

    vocab_path = opt.save_data + '.vocab.pkl'
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    return vocab


def _get_parser():
    parser = configargparse.ArgumentParser(description="preprocess.py")
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add('--data_dir', '-data_dir', required=True,
              help="Directory to the all data")
    group.add('--train_path', '-train_path', required=True,
              help="Path to the training data")
    group.add('--valid_path', '-valid_path', required=True,
              help="Path to the validation data")
    group.add('--save_data', '-save_data', required=True,
              help="Output file for the prepared data")
    return parser

if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    for key, value  in vars(opt).items():
        print("{0} = {1}".format(key, value))
    main(opt)
