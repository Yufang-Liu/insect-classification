#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/06/26 20:08:55

@author: Changzhi Sun
"""
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_neurons, num_classes, dropout=0.5):
        super(MLP, self).__init__()

        self.hidden_neurons = hidden_neurons
        
        self.input_fc = nn.Linear(input_dim, hidden_neurons[0])
        self.fcs = nn.ModuleList([nn.Linear(hidden_neurons[i], hidden_neurons[i+1]) for i in range(len(hidden_neurons)-1)])
        self.output_fc = nn.Linear(hidden_neurons[-1], num_classes)
        self.drop = nn.Dropout(p=dropout)
        
    def forward(self, x):
        
        #flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_fc(x))
        x = self.drop(x)
        for i in range(len(self.hidden_neurons)-1):
            x = F.relu(self.fcs[i](x))
            x = self.drop(x)
        x = self.output_fc(x)
        return x

def conv2d_same(length, width, in_channels, out_channels, kernel_size, stride=1,
                dilation=1, groups=1, bias=True, padding_mode='zeros'):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    kernel_size = np.array(kernel_size)
    stride = np.array(stride)
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    input_size = np.array([length, width])

    output_size = (input_size + stride - 1) // stride
    #  print("output size:", output_size)
    #  print("effective_kernel_size:", effective_kernel_size)
    tmp = (output_size - 1) * stride + effective_kernel_size - input_size

    zeros = np.array([0, 0])
    #  padding_needed = np.stack([np.array([0, 0]), tmp]).max(0)
    padding_needed = np.where(tmp > zeros, tmp, zeros)
    padding_before = padding_needed // 2
    padding_after = padding_needed - padding_before
    #  print(padding_after)
    return nn.Conv2d(in_channels, out_channels, tuple(kernel_size), tuple(stride), padding=tuple(padding_after),
                     dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

class CNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(CNN, self).__init__()

        self.conv1 = conv2d_same(200, 310, 1, 96, (16, 16), (8, 8))
        self.conv2 = conv2d_same(12, 19, 96, 256, (5, 3), (1, 1))
        self.conv3 = conv2d_same(6, 9, 256, 384, (3, 3), (1, 1))
        self.conv4 = conv2d_same(6, 9, 384, 384, (3, 3), (1, 1))
        self.pool = nn.MaxPool2d(2, 2)

        self.drop = nn.Dropout(p=dropout)

        self.conv = nn.Sequential(
                self.conv1,
                nn.ReLU(True),
                self.pool,
                self.drop,
                self.conv2,
                nn.ReLU(True),
                self.pool,
                self.drop,
                self.conv3,
                nn.ReLU(True),
                self.conv4,
                nn.ReLU(True),
                self.pool,
                self.drop,
            )

        self.mlp = nn.Sequential(
                nn.Linear(4608, 2048),
                nn.ReLU(True),
                self.drop,
                nn.Linear(2048, 2048),
                nn.ReLU(True),
                self.drop
            )
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 1, 200, 310)
    conv = conv2d_same(200, 310, 1, 96, (16, 16), (8, 8))
#  print(conv(x).size())
