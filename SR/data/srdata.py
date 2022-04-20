import os

from data import common

import numpy as np
import scipy.misc as misc
import imageio
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)   # 对于div2k来说dir_hr，dir_lr，ext

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin(), allow_pickle=True)
            self.images_lr = [
                np.load(self._name_lrbin(s), allow_pickle=True) for s in self.scale
            ]


        """
        1. benchmark --> 测试集 或者 args.ext == 'img'
            获得路径
        2. args.ext.find('sep')     文件分开存储为npy格式
            sep                     读npy格式的文件
            sep_reset               转换文件格式+读npy格式的文件
        3. args.ext.find('bin')     所有图片文件存成一个文件
            bin                     读npy格式的文件（一个）
            bin_reset               转换文件格式+读npy格式的文件
        """
        if args.ext == 'img' or benchmark:    # benchmark --> 测试集 或者 img
            self.images_hr, self.images_lr = self._scan()   # 获得所有图片路径
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = self._scan()   # 获得所有图片路径
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:                    # hr 循环转换 .png --> .npy
                    hr = imageio.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):         # hr 循环转换 .png --> .npy
                    for v in self.images_lr[si]:
                        lr = imageio.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [                              # 新的 list 存储 hr 数据路径
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [                              # 新的 list 存储 lr 数据路径
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')    # self.apath = dir_data + '/DIV2K800'
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr = self._scan()
                hr = [imageio.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)               # hr 是个 list 存储了所有的图片信息
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [imageio.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load_bin()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):                     # dataset标准
        lr, hr, filename = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = imageio.imread(lr)
            hr = imageio.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

