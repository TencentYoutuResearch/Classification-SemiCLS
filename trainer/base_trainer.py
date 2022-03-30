""" The Code is under Tencent Youtu Public Rule
This file set the base trainer format for the framework
To write your own SSL althorithm, please inherit base trainer as a base class
"""
from abc import ABCMeta, abstractmethod


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def compute_loss(self):
        pass

    def get_pseudo_acc(self, p_targets, targets, mask=None):
        if mask is not None:
            right_labels = (p_targets == targets).float() * mask
            return right_labels.sum() / max(mask.sum(), 1.0)
        else:
            right_labels = (p_targets == targets).float()
            return right_labels.mean()
