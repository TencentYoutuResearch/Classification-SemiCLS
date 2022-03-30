""" The Code is under Tencent Youtu Public Rule
Builder for scheduler
"""
from copy import deepcopy
from functools import partial

from .cosine_with_warmup import cosine_schedule_with_warmup

scheduler_dict = {"cosine_schedule_with_warmup": cosine_schedule_with_warmup}


def build(cfg):
    scheduler_cfg = deepcopy(cfg)
    type_name = scheduler_cfg.pop("type")
    return partial(scheduler_dict[type_name], **scheduler_cfg)
