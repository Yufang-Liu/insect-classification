#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/07/10 05:41:53

@author: Changzhi Sun
"""
import os
import pickle
import numpy as np
import random
from collections import Counter, defaultdict
import configargparse


def main(opt):

    label_ct = defaultdict(list)
    pkl_label_path = os.path.join(opt.data_dir, "pkl_label.txt")
    with open(pkl_label_path, "r") as f:
        for line in f:
            assert len(line.strip().split('\t')) == 2
            _, label = line.strip().split('\t')
            label_ct[label].append(_)
    train_path = os.path.join(opt.data_dir, "train.txt")
    valid_path = os.path.join(opt.data_dir, "valid.txt")
    ftrain = open(train_path, "w", encoding="utf8")
    fvalid= open(valid_path, "w", encoding="utf8")
    for k, v in label_ct.items():
        if len(v) > 1:
            num_valid = int(len(v) * 0.2)
            if num_valid == 0:
                num_valid = 1
            np.random.shuffle(v)
            valid_pkl = v[:num_valid]
            train_pkl = v[num_valid:]
            for e in train_pkl:
                #  with open(os.path.join("data", e), "rb") as f:
                    #  mat = pickle.load(f)
                #  print(np.isnan(mat).sum())
                print("%s\t%s" % (e, k), file=ftrain)
            for e in valid_pkl:
                print("%s\t%s" % (e, k), file=fvalid)
        elif len(v) == 1:
            pkl_path = os.path.join(opt.data_dir, v[0])
            with open(pkl_path, "rb") as f:
                mat = pickle.load(f)
            num_valid = int(len(mat) * 0.2)
            if num_valid == 0:
                num_valid = 1
            valid_mat = mat[-num_valid:]
            train_mat = mat[:-num_valid]
            
            pkl_dir = os.path.dirname(pkl_path)
            basename = os.path.basename(pkl_path)
            train_pkl_path = os.path.join(pkl_dir, "train.%s" % basename)
            valid_pkl_path = os.path.join(pkl_dir, "valid.%s" % basename)
            with open(train_pkl_path, "wb") as f:
                pickle.dump(train_mat, f)

            with open(valid_pkl_path, "wb") as f:
                pickle.dump(valid_mat, f)

            print(train_pkl_path)
            print(valid_pkl_path)
            train_pkl_path = '/'.join(train_pkl_path.split("/")[-4:])
            valid_pkl_path = '/'.join(valid_pkl_path.split("/")[-4:])
            print(train_pkl_path)
            print(valid_pkl_path)
            print("%s\t%s" % (train_pkl_path, k), file=ftrain)
            print("%s\t%s" % (valid_pkl_path, k), file=fvalid)

    ftrain.close()
    fvalid.close()

def _get_parser():
    parser = configargparse.ArgumentParser(description="evaluation.py")

    group = parser.add_argument_group('Data')
    group.add('--data_dir', '-data_dir', required=True,
              help="Directory to the all data")
    return parser

if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    for key, value  in vars(opt).items():
        print("{0} = {1}".format(key, value))
    print()
    np.random.seed(1111)
    main(opt)
