# coding=utf-8
import numpy as np
import json
import math
import os
import pickle
import re
import sys
import time
import matplotlib.pyplot as plt
import configparser  # https://www.cnblogs.com/dion-90/p/7978081.html 读写ini
from scipy import optimize
plt.rcParams.update({'font.size': 16})
from sklearn import preprocessing

#nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module, Parameter, init
from torch.optim import lr_scheduler
from collections import OrderedDict

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def list_to_torch(list):
    tensor = to_torch(np.array(list)).float()
    return tensor

def get_word_embedding(root_path):
    try:
        return np.load(root_path+"word_embedding.npy")
    except FileNotFoundError:
        with open(root_path+"deepwalk_128_unweighted_with_args.txt") as f:
            index_embedding = {}
            for line in f:
                line = line.strip().split()
                if len(line) == 2:
                    continue
                index_embedding[line[0]] = np.array(line[1:], dtype=np.float32)
            index_embedding["0"] = np.zeros(len(index_embedding["0"]), dtype=np.float32)
        word_embedding = []
        for i in range(len(index_embedding)):
            word_embedding.append(index_embedding[str(i)])
        word_embedding = np.array(word_embedding, dtype=np.float32)
        np.save(root_path+"word_embedding", word_embedding)
        return word_embedding
