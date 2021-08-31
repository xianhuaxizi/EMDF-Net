#!/usr/bin/env python
# coding: utf-8

import os
import time
import json
from collections import OrderedDict
import importlib
import logging
import argparse
import numpy as np
import random
import scipy.misc
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
import torch.backends.cudnn
from tqdm import tqdm


import src.datasources as datasources
import src.models as models
from config import *
from src.util.osutils import mkdir_p, isfile, isdir, join
from src.util.tools import *


torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def inference(test_index_name, test_file_name, model_path, root_path, gpu_device):
    if not isfile(model_path):
        logger.info("=> no checkpoint found at '{}'".format(model_path))

    if torch.cuda.is_available() and gpu_device is not None:
        device = torch.device(gpu_device)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

    logger.info("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    cfg = checkpoint['cfg']
    new_config = {'isTrain': False, 'device': gpu_device, 'root_path': root_path}
    cfg.parse(new_config)
    test_index = pickle.load(open(cfg.root_path + test_index_name, "rb"))

    # create model
    logger.info("==> creating model '{}'".format(cfg.model_arch))
    model = models.__dict__[cfg.model_arch](cfg)
    model.load_networks(checkpoint)
    model.to(device)
    model.print_networks(False)

    test_dataset = datasources.ScriptData(data_path=cfg.root_path+test_file_name, one=True, direct=True)
    test_data = test_dataset.all_data()
    logger.info('Totally %d samples' % (len(test_dataset)))

    # Start!
    logger.info("Start testing!\n")
    model.eval()
    start = time.time()
    with torch.no_grad():
        model.set_input(test_data, device)

        test_acc_list = model.process_test_all(test_index)
        print("Test Acc_level1: %f" % test_acc_list[0])
        print("Test Acc_level2: %f" % test_acc_list[1])
        print("Test Acc_level3: %f" % test_acc_list[2])
        print("Test Acc_level_feature: %f" % test_acc_list[3])
        print("Test Acc_level_score: %f" % test_acc_list[4])

    end = time.time()
    avg_time = (end - start) / len(test_dataset)
    logger.info("average time is %f seconds" % avg_time)


if __name__ == '__main__':
    test_index_name = 'test_index.pickle'
    test_file_name = 'vocab_index_test.data'
    root_path = "../../data/"
    model_path ='../models/EMDF-Net_best_acc_69.59.pth'

    gpu_device = "cuda:2" # None if set cpu

    inference(test_index_name, test_file_name, model_path, root_path, gpu_device)
