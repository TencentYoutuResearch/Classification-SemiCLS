""" The Code is under Tencent Youtu Public Rule
build optimizer:
optimizer_dict = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW
}

"""
from copy import deepcopy

import torch.optim as optim

from .lars_optimizer import LARS

optimizer_dict = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW
}


def build(cfg, model):
    optimizer_cfg = deepcopy(cfg)
    optim_type = optimizer_cfg.pop("type")

    use_lars = False
    if "lars" in optimizer_cfg.keys():
        use_lars = optimizer_cfg.pop("lars")

    no_decay = optimizer_cfg.pop("no_decay", ['bias', 'bn'])
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)],
            'weight_decay': optimizer_cfg.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optimizer_dict[optim_type](grouped_parameters, **optimizer_cfg)

    if use_lars:
        optimizer = LARS(optimizer)
    return optimizer
