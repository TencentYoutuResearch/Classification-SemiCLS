""" The Code is under Tencent Youtu Public Rule
This file build the meta archs for the framework
Each training must register here
"""

from copy import deepcopy
from functools import partial

from .classifier import Classifier
from .comatch import CoMatch
from .fixmatch import FixMatch
from .fixmatch_ccssl import FixMatchCCSSL

# meta archs for all trainers
meta_archs = {
    "FixMatch": FixMatch,
    "CoMatch": CoMatch,
    "Classifier": Classifier,
    "FixMatchCCSSL": FixMatchCCSSL,
}


def build(cfg):
    """ build function for trainer
        the cfg must contain type for trainer
        other configs will be used as parameters
    """
    trainer_cfg = deepcopy(cfg)
    type_name = trainer_cfg.pop("type")
    return partial(meta_archs[type_name], cfg=trainer_cfg)
