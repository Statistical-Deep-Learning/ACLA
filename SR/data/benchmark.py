import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_hr):    # 目录迭代方法
            filename = os.path.splitext(entry.name)[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                    # '/data/yyang409/yancheng/data/SR' + 'benchmark' +
                    # self.args.data_test ('Set5') + LR_bicubic + X2/namex2.png
                ))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        # '/data/yyang409/yancheng/data/SR' + 'benchmark' + self.args.data_test ('Set5')
        self.dir_hr = os.path.join(self.apath, 'HR')
        # '/data/yyang409/yancheng/data/SR' + 'benchmark' + self.args.data_test ('Set5') + HR
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        # '/data/yyang409/yancheng/data/SR' + 'benchmark' + self.args.data_test ('Set5') + LR_bicubic
        self.ext = '.png'
