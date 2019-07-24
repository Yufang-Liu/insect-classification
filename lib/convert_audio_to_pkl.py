#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/07/22 10:49:20

@author: Changzhi Sun
"""
import os
import sys
import pickle
sys.path.append("..")

import configargparse 
from lib.interface import sound2numpy

def main(opt):
    all_data_dir = os.path.join(opt.data_dir, "all_data")
    fa = open(os.path.join(opt.data_dir, "pkl_label.txt"), "w")
    for date in os.listdir(all_data_dir):
        pkl_label_path = os.path.join(all_data_dir, date, "pkl_label.txt")
        if os.path.exists(pkl_label_path):
            with open(pkl_label_path, "r", encoding="utf8") as t:
                for line in t:
                    print("all_data/%s/sound/" % date + line.strip(), file=fa)
            continue

        label_path = os.path.join(all_data_dir, date, "label.txt")
        with open(label_path, "r", encoding="utf8") as f:
            with open(pkl_label_path, "w", encoding="utf8") as fp:
                for line in f:
                    audio_path = line.strip().split('\t')[0]
                    label_name = line.strip().split('\t')[1]
                    
                    audio_path = os.path.join(all_data_dir, date, "sound", audio_path)
                    pkl_path = os.path.join(
                        os.path.dirname(audio_path),
                        os.path.basename(audio_path).split('.')[0] + ".pkl")
                    mat = sound2numpy(audio_path)
                    with open(pkl_path, "wb") as g:
                        pickle.dump(mat, g)
                    print("{0}\t{1}".format(os.path.basename(pkl_path), label_name), file=fp)
                    save_path = "/".join(pkl_path.split('/')[-4:])
                    print("{0}\t{1}".format(save_path, label_name), file=fa)
    fa.close()


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
    main(opt)
