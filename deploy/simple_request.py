#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/07/17 14:18:36

@author: Changzhi Sun
"""
import configargparse
import requests
import sys
sys.path.append("..")
import numpy as np
import torch
import pickle
from lib.interface import sound2numpy

def predict_result(opt):
    # Initialize image path

    #  mat = torch.randn(3, 200, 310).numpy()

    #  with open("../../2.pkl", "rb") as f:
        #  mat = pickle.load(f)
    mat = sound2numpy(opt.audio_path)
    mat = mat.astype(np.float32)

    payload = {'mat': mat, 'top_num': opt.top_num}
    # Submit the request.
    full_url = opt.rest_api_url + "/" + opt.version
    r = requests.post(full_url, files=payload).json()

    # Ensure the request was successful.
    print()
    print()
    if r['success']:
        # Loop over the predictions and display them.
        for (i, result) in enumerate(r['results']):
            print('{}. {}: {:.4f}'.format(i + 1, result['name'],
                                          result['score']))
    # Otherwise, the request failed.
    else:
        print('Request failed')
        if 'help' in r:
            print(r['help'])

def _get_parser():
    parser = configargparse.ArgumentParser(description="simple_request.py")
    group = parser.add_argument_group('Data')
    group.add('--audio_path', '-audio_path', required=True,
              help="Path to audio file")

    group = parser.add_argument_group('Server')
    group.add('--rest_api_url', '-rest_api_url', type=str,
              default="http://127.0.0.1:5000/predict",
              help="Initialize the PyTorch REST API endpoint URL")
    group.add('--top_num', '-top_num', type=int,
              default=2,
              help="Return top predictions")
    group.add('--version', '-version', type=str, default="v0",
                help="The version of model")
    return parser

if __name__ == '__main__':
    parser = _get_parser()
    opt = parser.parse_args()
    for key, value  in vars(opt).items():
        print("{0} = {1}".format(key, value))
    print()
    predict_result(opt)
