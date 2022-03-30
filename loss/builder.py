""" The Code is under Tencent Youtu Public Rule
This file builds the loss term for the framework
"""
from copy import deepcopy
from functools import partial
import torch
import torch.nn.functional as F

loss_dict = {"cross_entropy": F.cross_entropy}


def build(cfg):
    loss_cfg = deepcopy(cfg)
    loss_type = loss_cfg.pop("type")
    return partial(loss_dict[loss_type], **loss_cfg)
