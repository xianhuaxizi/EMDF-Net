import os
import torch
import torchvision.utils
from collections import OrderedDict
from abc import ABC, abstractmethod
from config import *
from src.util.osutils import mkdir_p, isfile, isdir, join
from src.util.tools import *


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
        """
        self.isTrain = opt.isTrain
        self.acc_names = []
        self.model_names = []
        self.output_names = []


    @abstractmethod
    def set_input(self, input, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
            device:  指定设备
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def eval(self):
        """Make models eval mode during test/val time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def to(self, device):
        """迁移模型到device"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.to(device)

    def load_networks(self, checkpoint):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if name in checkpoint.keys():
                    net.load_state_dict(checkpoint[name])

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
