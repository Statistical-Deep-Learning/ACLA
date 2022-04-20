import os
import math
import time
import datetime
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLR_Restart(_LRScheduler):
    """
    ref:https://github.com/zhaohengyuan1/PAN/blob/a20974545cf011c386d728739d091c39e23d0686/codes/models/lr_scheduler.py
    ref: pytorch_CosineAnnealingLR_doc  https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
    """

    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type == 'cosine':

        ##############################
        # Cosine Annealing Restart
        ##############################

        ## need more to prevent error
        # T_period = [250, 500, 750, 1000, 1250]
        # restarts = [250, 500, 750, 1000, 1250]
        T_period = list(range(args.cosine_restart_epoch,
                              int((args.epochs / args.cosine_restart_epoch + 1) * args.cosine_restart_epoch),
                              args.cosine_restart_epoch))
        restarts = T_period
        restart_weights = [1 for _ in range(len(restarts))]

        scheduler = CosineAnnealingLR_Restart(my_optimizer, T_period, eta_min=1e-7, restarts=restarts,
                                              weights=restart_weights)
    return scheduler