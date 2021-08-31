# coding=utf-8
import configparser
from glob import glob
import os
import random
import time
from math import ceil

from src.util.tools import *
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式


def my_collate_fn(batch):
    # 过滤为None的数据
    batch = list(filter(lambda x: x['event_chain'] is not None, batch))
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


class ScriptData(data.Dataset):
    def __init__(self,
                 data_path='data/metadata/vocab_index_train.data',
                 one=True,
                 direct=True
                 ):
        # Create global index over all specified keys
        self._data_path = data_path
        self._data = pickle.load(open(self._data_path, "rb"))
        self.adj_matrix = self._data[0]
        self.event_chain = self._data[1]
        self.label = self._data[2]
        if one:
            self.adj_matrix = self.trans_to_one(self.adj_matrix, self.label)
        if not direct:
            self.adj_matrix = self.trans_to_undirected(self.adj_matrix)

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


    def all_data(self, index=None):
        if index is None:
            entry = {
                'event_chain': self.event_chain,
                'adj_matrix': self.adj_matrix,
                'label': self.label
            }
        else:
            entry = {
                'event_chain': self.event_chain.index_select(0, index),
                'adj_matrix': self.adj_matrix.index_select(0, index),
                'label': self.label.index_select(0, index)
            }
        return entry

    def __getitem__(self, item):
        _event_chain, _adj_matrix, _label = self.event_chain[item], self.adj_matrix[item], self.label[item]
        entry = {
            'event_chain': _event_chain,
            'adj_matrix': _adj_matrix,
            'label': _label
        }
        return entry

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    train_dataset = ScriptData(data_path='/home/zpp/PycharmProjects/Script-Event-Prediction_WWW21/data/metadata/vocab_index_train.data')

    val_dataset = ScriptData(data_path='/home/zpp/PycharmProjects/Script-Event-Prediction_WWW21/data/metadata/vocab_index_dev.data')

    training_data_loader = DataLoader(train_dataset, 6, collate_fn=my_collate_fn, shuffle=False, num_workers=0)

    val_data_loader = DataLoader(val_dataset, 6, collate_fn=my_collate_fn, shuffle=False, num_workers=0)

    dev_data = val_dataset.all_data()
    # cpu or gpu?
    device_gpu = "cuda:2"
    if torch.cuda.is_available() and device_gpu is not None:
        device = torch.device(device_gpu)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")
    for ii, batch in enumerate(training_data_loader):
        event_chain = batch['event_chain'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)
        label = batch['label'].to(device)  # [0,1] 归一化

    for ii, batch in enumerate(val_data_loader):
        event_chain = batch['event_chain'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)
        label = batch['label'].to(device)

