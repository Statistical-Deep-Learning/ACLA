import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class DIV2K(srdata.SRData):
    """继承自 srdata 的类

    """
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)
        # test_every: do test per every N batches
        # n_train: number of training set
        self.repeat = args.test_every // (args.n_train // args.batch_size) # 200 // (800 // 16) = 200 // 50iter = 4

    def _scan(self):
        """
        Return:
            list_hr: a list of hr image path
            list_lr: a list of lr image path [[scale1_pic1, scale1_pic2, scale1_pic3,...], [scale2_pic1, scale2_pic2, scale2_pic3,...], ...]
        """

        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:    
            """训练集0001-0800
            DIV2K/DIV2K_train_HR/ -- 0001.png, 0002.png, ..., 0800.png train HR images 
            (provided to the participants)
            """
            idx_begin = 0
            idx_end = self.args.n_train
        else:                                       
            """在DIV2K上测试的话，测试集801-900
            DIV2K提供了测试集，测试集的数量801-》
            [DIV2K_website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
            DIV2K/DIV2K_valid_HR/ -- 0801.png, 0802.png, ..., 0900.png validation HR images 
            (will be available to the participants at the beginning of the final evaluation phase)
            """
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val   # offset_val  default 800

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)   # i=2-->0002;   i=123-->0123
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))   # ext 扩展名
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}.00/{}{}'.format(s, filename,  self.ext)  # img 的话会被自动转换成 .png 见下面
                ))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/DIV2K' 
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = '.png'

    def _name_hrbin(self):                    # 存储了所有的图片的 .npy 文件
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):             # 存储了所有的图片的 .npy 文件
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        """
        虚拟出一个epoch，之前是 iter 个 batch_size 之后结束一个 epoch ，然后valid在 test_every 个 iter 之后测试
        现在将一个 epoch 虚拟为 n_train * repeat =  n_train * (test_every // (n_train // batch_size))  
        800 * (200 // (800 // 16) = 200 // 50iter = 4) = 3200
        """
        if self.train:
            return len(self.images_hr) * self.repeat   
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

