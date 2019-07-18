#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/06/26 17:02:09

@author: Changzhi Sun
"""
import os
import pickle

import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class ImgDataset(Dataset):
    def __init__(self, data_dir: str, label_path: str, vocab: dict):
        self.data_dir = data_dir
        self.vocab = vocab
        self.data = []
        self.distributions = [0]

        n_files = self.get_line_number(label_path)
        with open(label_path, "r", encoding="utf8") as f:
            for i, line in tqdm(enumerate(f), total=n_files):
                assert len(line.split('\t')) == 2
                pkl_name, label_str = line.strip().split('\t')
                label = vocab[label_str]

                pkl_path = os.path.join(data_dir, pkl_name)
                with open(pkl_path, "rb") as g:
                    # (num_sample, width, length)
                    mat_samples = pickle.load(g)
                for mat in mat_samples:
                    mat = torch.FloatTensor(mat)
                    self.data.append((mat, label))
                self.distributions.append(len(self.data))
                
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_line_number(self, label_path):
        with open(label_path, "r") as f:
            ct = 0
            for line in f:
                ct += 1
            return ct
