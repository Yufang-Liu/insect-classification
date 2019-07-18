#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 19/06/26 19:09:16

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
import torch.nn.functional as F
import numpy as np

from src.model import MLP, CNN

def save_checkpoint(model, optim, vocab, base_path, i, j, val_score, is_best=False):
    real_model = (model.module if isinstance(model, nn.DataParallel) else model)
    checkpoint = {
        'model': real_model.state_dict(),
        'optim': optim.state_dict(),
        "vocab": vocab,
        "epoch": i,
        "val_score": val_score,
    }
    if is_best:
        print("Saving checkpoint %s_best.pt" % (base_path))
        best_checkpoint_path = '%s_best.pt' % (base_path)
        torch.save(checkpoint, best_checkpoint_path)
    else:
        print("Saving checkpoint %s_epoch_%d_step_%d.pt" % (base_path, i, j))
        checkpoint_path = '%s_epoch_%d_step_%d.pt' % (base_path, i, j)
        torch.save(checkpoint, checkpoint_path)

def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

def build_optimizer(model, opt, checkpoint):
    if opt.fine_tune and opt.fine_tune_only_linear:
        print("Only finetune linear parameters")
        parameters = model.linear.parameters()
    else:
        print("Update all parameters")
        parameters = model.parameters()
    if opt.optim == "adam":
        print('Building adam optimizer...')
        optim = torch.optim.Adam(
            parameters,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2))
    elif opt.optim == "noam":
        print('Building noam optimizer...')
        optim = NoamOpt(
            model.src_embed[0].d_model, 2,
            warmup=opt.warmup_steps,
            optimizer=torch.optim.Adam(parameters,
                lr=opt.learning_rate, betas=(opt.adam_beta1, opt.adam_beta2)))
    elif opt.optim == "adadelta":
        print('Building adadelta optimizer...')
        optim = torch.optim.Adadelta(parameters)

    if checkpoint is not None and opt.train_from:
        print("Loading optimizer parameters from checkpoint...")
        optim.load_state_dict(checkpoint['optim'])
    return optim

def build_dataset_iter(corpus_type, opt, is_train):
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    dataset = torch.load(opt.data + '.%s.pt' % corpus_type)
    print("%s data size: %d" % (corpus_type, len(dataset)))

    dataset_iter = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train)
    return dataset, dataset_iter


def build_model(opt, vocab, checkpoint, gpu_id=None):
    print('Building model...')
    use_gpu = len(opt.gpu_ranks) > 0
    if use_gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif use_gpu and not gpu_id:
        device = torch.device("cuda")
    elif not use_gpu:
        device = torch.device("cpu")

    #  model = MLP([1000, 500, 500], len(vocab))
    model = CNN(len(vocab), opt.dropout)

    if checkpoint is not None:
        # end of patch for backward compatibility
        if opt.train_from:
            print("Loading model parameters from checkpoint...")
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            print("Loading model parameters from checkpoint...")
            cur_state_dict = model.state_dict()                                 
            for key in cur_state_dict.keys():                                   
                if key in checkpoint['model'] and cur_state_dict[key].size() == checkpoint['model'][key].size():
                    cur_state_dict[key] = checkpoint['model'][key]              
                #  elif key in checkpoint['model'] and cur_state_dict[key].size() != checkpoint['model'][key].size():
                    #  print("***" , key)                              
            model.load_state_dict(cur_state_dict, strict=False)
    if len(opt.gpu_ranks) > 1:
        model = nn.DataParallel(model, opt.gpu_ranks)
    model.to(device)
    return model, device

def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def train_step(model, batch, criterion, optim, device):
    model.train()
    optim.zero_grad()

    x, y = batch
    x = x.to(device)
    y = y.to(device)

    logit = model(x)
    loss = criterion(logit, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
    optim.step()
    return loss

def get_metric(logit_array, y_array, n=1):
    metric = {"P@%d" % t : 0.0 for t in range(1, n+1)}
    AP_list = []
    logit_sort_dec = np.argsort(-logit_array, axis=1)
    for k in range(1, n+1):
        logit_sort_dec_k = logit_sort_dec[:, :k]
        y_in_k = [int(y in logit) for y, logit in zip(y_array, logit_sort_dec_k)]
        metric["P@%d" % k] = np.sum(y_in_k) / len(y_in_k)
    for i in range(len(y_array)):
        y = y_array[i]
        y_pos = list(logit_sort_dec[i]).index(y) + 1
        AP_list.append(1/y_pos)
    metric["MAP"] = np.mean(AP_list)
    return metric

def convert_prob_to_audio_index(probs, distributions):
    audio_probs = []
    for i in range(len(distributions) - 1):
        start, end = distributions[i], distributions[i+1]
        cur_audio_probs = np.mean(probs[start : end], axis=0)
        audio_probs.append(np.expand_dims(cur_audio_probs, 0))
    audio_probs = np.concatenate(audio_probs, 0)
    return audio_probs

def convert_label_to_audio_index(labels, distributions):
    true_labels = []
    for i in range(len(distributions) - 1):
        start, end = distributions[i], distributions[i+1]
        for j in range(start, end):
            assert labels[j] == labels[start]
        true_labels.append(labels[start])
    return np.array(true_labels)

def validation_step(model, valid_iter, valid_dataset, opt, device):
    model.eval()
    all_probs = []
    all_true_labels = []
    for i, batch in enumerate(valid_iter):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        p = F.softmax(model(x), dim=1)
        all_probs.append(p.cpu().detach().numpy())
        all_true_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs, 0)
    all_true_labels = np.concatenate(all_true_labels, 0)

    convert_all_probs = convert_prob_to_audio_index(
        all_probs,
        valid_dataset.distributions)
    convert_all_true_labels = convert_label_to_audio_index(
        all_true_labels, 
        valid_dataset.distributions)

    metric = get_metric(convert_all_probs, convert_all_true_labels, 100)
    return metric, len(all_true_labels)

def main(opt, device_id):
    configure_process(opt, device_id)
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        print('Loading vocab from checkpoint at %s' % opt.train_from)
        vocab = checkpoint['vocab']
        start_epoch = checkpoint['epoch'] + 1
        val_score = checkpoint['val_score']
        print('Epoch of checkpoint is %d' % checkpoint['epoch'])
        print('Validaton score of checkpoint is %f' % checkpoint['val_score'])
    elif opt.fine_tune:
        print('Loading checkpoint from %s' % opt.fine_tune)
        checkpoint = torch.load(opt.fine_tune,
                                map_location=lambda storage, loc: storage)
        print('Loading vocab from %s' % (opt.data + '.vocab.pkl'))
        with open(opt.data + '.vocab.pkl', "rb") as f:
            vocab = pickle.load(f)
        start_epoch = 0
        val_score = 0.0
    else:
        checkpoint = None
        print('Loading vocab from %s' % (opt.data + '.vocab.pkl'))
        with open(opt.data + '.vocab.pkl', "rb") as f:
            vocab = pickle.load(f)
        start_epoch = 0
        val_score = 0.0

    #  print(vocab)
    print(' * vocab size = %d' % (len(vocab)))
    model, device = build_model(opt, vocab, checkpoint)
    print(model)

    _check_save_model_path(opt)
    optim = build_optimizer(model, opt, checkpoint)
    
    print("Building training data iterator...")
    _, train_iter = build_dataset_iter("train", opt, is_train=True)
    print("Building validation data iterator...")
    valid_dataset, valid_iter = build_dataset_iter("valid", opt, is_train=False)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    if len(opt.gpu_ranks):
        print('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        print('Starting training on CPU, could be very slow')

    best_score = val_score
    patience_sum = 0
    for i in range(start_epoch, opt.epochs):
        start = time.time()
        num_instance = 0
        for j, batch in enumerate(train_iter):
            loss = train_step(model, batch, criterion, optim, device)
            num_instance += len(batch)
            if j > 0 and  j % 10 == 0:
                elapsed = time.time() - start
                print("Epoch: %d Step: %d Loss: %f Instances per Sec: %f" % (i, j, loss.item(), num_instance / elapsed))
                start = time.time()
                num_instance = 0  
        print("Evaluating validation set...")
        start = time.time()
        metric, total_ins = validation_step(model, valid_iter, valid_dataset, opt, device)
        elapsed = time.time() - start
        print("  * Epoch: %d Step: %d P@1  : %f Instances per Sec: %f" % (i, j, metric["P@1"], total_ins / elapsed))
        print("  * Epoch: %d Step: %d P@10 : %f Instances per Sec: %f" % (i, j, metric["P@10"], total_ins / elapsed))
        print("  * Epoch: %d Step: %d P@50 : %f Instances per Sec: %f" % (i, j, metric["P@50"], total_ins / elapsed))
        print("  * Epoch: %d Step: %d P@100: %f Instances per Sec: %f" % (i, j, metric["P@100"], total_ins / elapsed))
        print("  * Epoch: %d MAP  : %f Instances per Sec: %f" % (i, metric["MAP"], total_ins / elapsed))
        val_score = metric["MAP"]
        if val_score > best_score:
            best_score = val_score
            print("  * Achieving best score on validation set...")
            save_checkpoint(model, optim, vocab, opt.save_model, i, j, best_score, is_best=True)
            patience_sum = 0
        else:
            patience_sum += 1
        if opt.early_stopping_epochs != 0 and patience_sum == opt.early_stopping_epochs:
            break
        start = time.time()

def _get_parser():
    parser = configargparse.ArgumentParser(description="train.py")
    group = parser.add_argument_group('General')
    group.add('--data', '-data', required=True,
              help='Path prefix to the ".train.pt" and '
                   '".valid.pt" file path from preprocess.py')

    group.add('--save_model', '-save_model', default='model',
              help="Model filename (the model will be saved as "
                   "<save_model>_N.pt where N is the number "
                   "of steps")

    group.add('--save_checkpoint_steps', '-save_checkpoint_steps',
              type=int, default=5000,
              help="""Save a checkpoint every X steps""")
    group.add('--keep_checkpoint', '-keep_checkpoint', type=int, default=-1,
              help="Keep X checkpoints (negative: keep all)")

    # GPU
    group.add('--gpu_ranks', '-gpu_ranks', default=[], nargs='*', type=int,
              help="list of ranks of each process.")
    group.add('--seed', '-seed', type=int, default=1,
              help="Random seed used for the experiments "
                   "reproducibility.")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add('--train_from', '-train_from', default='', type=str,
              help="If training from a checkpoint then this is the "
                   "path to the pretrained model's state_dict.")
    group.add('--fine_tune', '-fine_tune', default='', type=str,
              help="If training from a checkpoint then this is the "
                   "path to the pretrained model's state_dict.")
    group.add('--fine_tune_only_linear', '-fine_tune_only_linear', default=False, type=bool,
              help="Only update last linear parameters")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add('--batch_size', '-batch_size', type=int, default=64,
              help='Maximum batch size for training')
    group.add('--valid_steps', '-valid_steps', type=int, default=1000,
              help='Perfom validation every X steps')
    group.add('--valid_batch_size', '-valid_batch_size', type=int, default=32,
              help='Maximum batch size for validation')
    group.add('--epochs', '-epochs', type=int, default=100,
              help='Number of epochs')
    group.add('--early_stopping_epochs', '-early_stopping_epochs', type=int, default=20,
              help='Number of validation epochs without improving.')
    group.add('--optim', '-optim', default='adam',
              choices=['adam', 'noam', 'adadelta'],
              help="Optimization method.")
    group.add('--max_grad_norm', '-max_grad_norm', type=float, default=5,
              help="If the norm of the gradient vector exceeds this, "
                   "renormalize it to have the norm equal to "
                   "max_grad_norm")
    group.add('--dropout', '-dropout', type=float, default=0.5,
              help="Dropout probability.")
    group.add('--adam_beta1', '-adam_beta1', type=float, default=0.9,
              help="The beta1 parameter used by Adam. "
                   "Almost without exception a value of 0.9 is used in "
                   "the literature, seemingly giving good results, "
                   "so we would discourage changing this value from "
                   "the default without due consideration.")
    group.add('--adam_beta2', '-adam_beta2', type=float, default=0.999,
              help='The beta2 parameter used by Adam. '
                   'Typically a value of 0.999 is recommended, as this is '
                   'the value suggested by the original paper describing '
                   'Adam, and is also the value adopted in other frameworks '
                   'such as Tensorflow and Kerras, i.e. see: '
                   'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
                   'Optimizer or '
                   'https://keras.io/optimizers/ . '
                   'Whereas recently the paper "Attention is All You Need" '
                   'suggested a value of 0.98 for beta2, this parameter may '
                   'not work well for normal models / default '
                   'baselines.')

    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add('--learning_rate', '-learning_rate', type=float, default=0.001,
              help="Starting learning rate. "
                   "Recommended settings: sgd = 1, adagrad = 0.1, "
                   "adadelta = 1, adam = 0.001")
    group.add('--warmup_steps', '-warmup_steps', type=int, default=4000,
              help="Number of warmup steps for custom decay.")

    return parser

if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    for key, value  in vars(opt).items():
        print("{0} = {1}".format(key, value))

    nb_gpu = len(opt.gpu_ranks)
    if nb_gpu >= 1:
        main(opt, opt.gpu_ranks[0])
    else:
        main(opt, -1)
