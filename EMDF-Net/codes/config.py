# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    def __init__(self):
        # Model structure
        self.model_arch = 'ScriptNet'  # 固定不变
        self.model_name = 'FusionNet'

        ## GCN
        self.positional_size = [36, 9]                       # sequence length, 8(context)+5(candidate), 8*4+4=36
        self.embedding_size = 128                       # size of argument
        self.hidden_size = self.embedding_size*4        # size of event
        self.n_layers = 1                               #
        self.n_heads = [4, 16]                       # 分别对应着不同层面的self-attention

        ## MLP Attention
        self.d_a = 256  # self.hidden_size/2            # {int} hyperparameter, the hidden_units of attention layer
        self.r = 2                                      # {int} attention-hops or attention heads, num of event segments
        self.penal_coeff = 0.03

        # Training strategy
        self.num_workers = 4      # 不使用多线程
        self.max_epochs = 100     # number of total epochs to run
        self.dropout = float(0.0)
        self.margin = float(0.05)
        self.lr = 2.0e-4      # initial learning rate
        self.weight_decay = 5e-4  # 损失函数
        self.momentum = 0.9
        self.train_batch = 2000  # train batchsize
        self.patients = int(5)  # Number of epochs with no improvement after which learning rate will be reduced
        self.schedule_lr = 10  #每10次下降一次学习率
        self.lr_decay = 0.4    # when val_loss increase, lr = lr*lr_decay  0.1
        self.lr_policy = 'plateau'


        self.log_interval = 10  # print log info and save history model every N epoch, if None,  log_interval = int(np.ceil(max_epochs * 0.02))
        self.betas = (0.9, 0.99)
        self.device = "cuda:2"

        # Data processing
        self.dataset = "NYT"
        self.root_path = "./data/metadata/" # localization of training set
        self.seed = 17  #随机数种子

        # Miscs
        self.checkpoint = "./checkpoints/"  # path to save checkpoint
        self.visible = True   # 是否显示中间结果 tensorboard

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

    def __str__(self):
        config = ""
        for name, value in vars(self).items():
            config += ('%s=%s\t\n' % (name, value))
        return config


if __name__ == '__main__':
    opt = DefaultConfig()
    new_config = {'lr': 0.1, 'device': "CPU"}
    opt.parse(new_config)
    print(opt)
