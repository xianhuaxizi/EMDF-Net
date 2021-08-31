import json
import math
import os
import pickle
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module, Parameter, init

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from sklearn import preprocessing


def compute_mask(positional_size):
    """
    Compute Mask matrix
    Mask: upper triangular matrix of masking subsequent information
        mask value: -1e9
    shape: (positional_size, positional_size)
    """
    return torch.triu(torch.fill_(torch.zeros(positional_size, positional_size), -1e9), 1).cuda()

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class InitLinear(Module):
    """
    Initialize Linear layer to be distribution function
    """
    def  __init__(self, inputs_size, outputs_size, dis_func, func_value, bias=True):
        super(InitLinear, self).__init__()
        self.outputs_size = outputs_size

        self.weight = Parameter(torch.empty(inputs_size, outputs_size), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(outputs_size), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters(dis_func, func_value)

    def reset_parameters(self, dis_func, func_value):
        if dis_func == "uniform":
            nn.init.uniform_(self.weight, -func_value, func_value)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -func_value, func_value)

        if dis_func == "normal":
            nn.init.normal_(self.weight, std=func_value)
            if self.bias is not None:
                nn.init.normal_(self.bias, std=func_value)

    def forward(self, inputs):
        output_size = inputs.size()[:-1] + (self.outputs_size,)
        if self.bias is not None:
            outputs = torch.addmm(self.bias, inputs.view(-1, inputs.size(-1)), self.weight)
        else:
            outputs = torch.mm(inputs.view(-1, inputs.size(-1)), self.weight)
        outputs = outputs.view(*output_size)
        return outputs


class SelfAttention(Module):
    """
    Self-Attention Layer

    Inputs:
        inputs: word embedding
        inputs.shape = (batch_size, sequence_length, embedding_size)

    Outputs:
        outputs: word embedding with context information
        outputs.shape = (batch_size, sequence_length, embedding_size)
    """
    def __init__(self, embedding_size, n_heads, dropout):
        super(SelfAttention, self).__init__()
        assert embedding_size % n_heads == 0
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.d_k = embedding_size // n_heads

        self.w_qkv = InitLinear(embedding_size, embedding_size*3, dis_func="normal", func_value=0.02)
        self.w_head = InitLinear(embedding_size, embedding_size, dis_func="normal", func_value=0.02)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, k=False):
        x = x.view(-1, x.size(1), self.n_heads, self.d_k)
        if k:
            return x.permute(0, 2, 3, 1)  # key.shape = (batch_size, n_heads, d_k, sequence_length)
        else:
            return x.permute(0, 2, 1, 3)  # query, value.shape = (batch_size, n_heads, sequence_length, d_k)

    def attention(self, query, key, value, mask=None):
        att = torch.matmul(query, key) / math.sqrt(self.d_k)

        if mask is not None:
            att = att + mask

        att = torch.softmax(att, -1)
        att = self.dropout(att)

        # att = self.sample(att)

        outputs = torch.matmul(att, value)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        x = self.dropout(self.w_head(x))
        return x

    @staticmethod
    def sample(att):
        att_ = att.view(-1, att.size(-1))
        _, tk = torch.topk(att_, k=26, largest=False)
        for i in range(att_.size(0)):
            att_[i][tk[i]] = 0.0
        att = att_.view(att.size())
        return att

    def forward(self, inputs, mask=None):
        inputs = self.w_qkv(inputs)
        query, key, value = torch.split(inputs, self.embedding_size, 2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)  # k=True，相当于进行了转置
        value = self.split_heads(value)

        att_outputs = self.attention(query, key, value, mask)
        outputs = self.merge_heads(att_outputs)
        return outputs

class EventComposition(Module):
    """
    Event Composition layer
        integrate event argument embedding into event embedding

    Inputs:
        inputs: arguments embedding
        inputs.shape = (batch_size, argument_length, embedding_size)

    Outputs:
        outputs: event embedding
        outputs.shape = (batch_size, event_length, hidden_size)
    """
    def __init__(self, inputs_size, outputs_size, dropout):
        super(EventComposition, self).__init__()
        self.outputs_size = outputs_size

        self.w_e = InitLinear(inputs_size*4, outputs_size, dis_func="normal", func_value=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = inputs.view(-1, inputs.size(1) // 4, self.outputs_size)
        outputs = self.dropout(torch.tanh(self.w_e(inputs)))
        return outputs


class Data:
    """
    Load data
    """
    def __init__(self, data, one=True, direct=True):
        self.adj_matrix = data[0]
        if one:
            self.adj_matrix = self.trans_to_one(self.adj_matrix, data[2])
        if not direct:
            self.adj_matrix = self.trans_to_undirected(self.adj_matrix)
        self.event_chain = data[1]
        self.label = data[2]
        self.len = len(self.label)
        self.start = 0
        self.flag_epoch = True

    @staticmethod
    def trans_to_one(matrix, label):
        matrix = torch.where(matrix > 0, torch.ones_like(matrix), matrix)
        for i in range(matrix.size(0)):
            for j in range(7):
                if matrix[i][j][j+1] == 0:
                    matrix[i][j][j+1] = 1
            if matrix[i][7][8+label[i]] == 0:  # 除了正确的答案以外，是否其他的候选答案也应该连接。
                matrix[i][7][8+label[i]] = 1
        return matrix

    @staticmethod
    def trans_to_undirected(matrix):
        return torch.add(matrix, matrix.permute(0, 2, 1))

    def next_batch(self, batch_size):
        start = self.start
        end = self.start + batch_size if self.start + batch_size < self.len else self.len
        self.start = self.start + batch_size
        if self.start < self.len:
            self.flag_epoch = True
        else:
            self.start = self.start % self.len
            self.flag_epoch = False
        return [to_cuda(self.event_chain[start: end]),
                to_cuda(self.adj_matrix[start: end]),
                to_cuda(self.label[start: end])]

    def all_data(self, index=None):
        if index is None:
            return [to_cuda(self.event_chain), to_cuda(self.adj_matrix), to_cuda(self.label)]
        else:
            return [to_cuda(self.event_chain.index_select(0, index)),
                    to_cuda(self.adj_matrix.index_select(0, index)),
                    to_cuda(self.label.index_select(0, index))]


def to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
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
