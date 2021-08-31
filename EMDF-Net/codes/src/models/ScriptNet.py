from .FusionNet import FusionNet
from .base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util.tools import *
from config import DefaultConfig
from sklearn import preprocessing
import pickle as pkl
import numpy as np

class ScriptNet(BaseModel):
    def __init__(self, cfg=DefaultConfig()):
        super(ScriptNet, self).__init__(cfg)
    #    BaseModel.__init__(self, cfg)

        self.word_embedding = get_word_embedding(cfg.root_path)

        if cfg.model_name == "FusionNet":
            self.model_H = FusionNet(vocab_size= len(self.word_embedding),
                           embedding_size=cfg.embedding_size,
                           word_embedding=self.word_embedding,
                           hidden_size= cfg.hidden_size,
                           dropout=cfg.dropout,
                           num_layers= cfg.n_layers,
                           bidirectional=False,
                           positional_size=cfg.positional_size,
                           n_heads= cfg.n_heads,
                           d_a = cfg.d_a,
                            r = cfg.r)
        else:
            return

        self.model_names = ['model_H']
        self.acc_names=['level1', 'level2', 'level3', 'level4', 'levelFusion']

    def set_input(self, input, device):
        self.event_chain = input['event_chain'].to(device)
        self.adj_matrix = input['adj_matrix'].to(device)
        self.label = input['label'].to(device)

    def forward(self):
        self.predict = self.model_H(self.event_chain, self.adj_matrix)
        self.outputs_level1 = self.predict[0]
        self.outputs_level2 = self.predict[1]
        self.outputs_level3 = self.predict[2]
        self.outputs_level4 = self.predict[3]
        self.outputs_fusion = self.predict[4]

    # calculate accuracy
    def predict_func(self, predict, label):
        _, predict = torch.sort(predict, descending=True)
        n_correct = torch.sum((predict[:, 0] == label)).item()
        n_label = label.size(0)
        acc = n_correct / n_label * 100.0
        return acc

    def process_test_all(self, outlier_index):
        self.forward()
        for index in outlier_index:
            self.outputs_level1[index] = -1e9
            self.outputs_level2[index] = -1e9
            self.outputs_level3[index] = -1e9
            self.outputs_level4[index] = -1e9
            self.outputs_fusion[index] = -1e9
        acc_level1 = self.predict_func(self.outputs_level1, self.label)
        acc_level2 = self.predict_func(self.outputs_level2, self.label)
        acc_level3 = self.predict_func(self.outputs_level3, self.label)
        acc_level4 = self.predict_func(self.outputs_level4, self.label)
        acc_levelFusion = self.predict_func(self.outputs_fusion, self.label)
        return (acc_level1, acc_level2, acc_level3, acc_level4, acc_levelFusion)

if __name__ == "__main__":
    cfg = DefaultConfig()
    net = ScriptNet(cfg)
    net.print_networks(True)

