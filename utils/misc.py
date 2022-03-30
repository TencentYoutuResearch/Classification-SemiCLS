""" The Code is under Tencent Youtu Public Rule

Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
code in this file is adpated from
https://github.com/kekmodel/FixMatch-pytorch/blob/master/utils/misc.py
thanks!
"""
import logging

import numpy as np
import torch
from torch.nn.functional import instance_norm

logger = logging.getLogger(__name__)

__all__ = [
    'get_mean_and_std', 'accuracy', 'AverageMeter', 'AverageMeterManeger'
]


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HistgramMeter(object):
    """ Compute histgram meter
    """
    def __init__(self, default_len=10000):
        self.default_len = default_len
        self.reset()

    def reset(self):
        self._data_list = []

    @property
    def data_list(self):
        return np.array(self._data_list)

    def update(self, value):
        if len(self._data_list) > self.default_len:
            self._data_list = self._data_list[-self.default_len:]
        self._data_list.extend(list(value))


class AverageMeterManeger(object):
    def __init__(self):
        self.name_list = []

    def register(self, name, value=0):
        self.name_list.append(name)
        if isinstance(value, np.ndarray):
            setattr(self, name, HistgramMeter())
        else:
            setattr(self, name, AverageMeter())

    def reset_avgmeter(self):
        for idx, key in enumerate(self.name_list):
            if isinstance(getattr(self, key), AverageMeter):
                getattr(self, key).reset()

    def try_register_and_update(self, reg_data):
        for key, value in reg_data.items():
            if not hasattr(self, key):
                self.register(key, value)

            if isinstance(value, torch.Tensor):
                getattr(self, key).update(value.item())
            else:
                getattr(self, key).update(value)

    def get_desc(self):
        meter_desc = ""
        for key in self.name_list:
            if isinstance( getattr(self, key), AverageMeter):
                meter_desc += "{}: {:.3f} ".format(key, getattr(self, key).avg)
        return meter_desc

    def add_to_writer(self, writer, epoch, prefix="train/"):
        for idx, key in enumerate(self.name_list):
            if isinstance(getattr(self, key), AverageMeter):
                writer.add_scalar("{}.{} ".format(prefix, key),
                                  getattr(self, key).avg, epoch)
            elif isinstance(getattr(self, key), HistgramMeter):
                writer.add_histogram("{}.{}".format(prefix, key),
                                     getattr(self, key).data_list, epoch)
            else:
                raise ValueError("Unsupported value type {}".format(
                    type(getattr(self, key))))
