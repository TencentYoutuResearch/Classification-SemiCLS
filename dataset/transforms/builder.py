""" The Code is under Tencent Youtu Public Rule
builder for transforms

transforms from torch or home-made
"""

import copy

from torchvision import transforms

from .randaugment import RandAugmentMC
from .gaussian_blur import GaussianBlur
other_func = {"RandAugmentMC": RandAugmentMC,"GaussianBlur":GaussianBlur}


def get_trans(trans_cfg):
    init_params = copy.deepcopy(trans_cfg)
    type_name = init_params.pop("type")
    if type_name in other_func.keys():
        return other_func[type_name](**init_params)
    if type_name == "RandomApply":
        r_trans = []
        trans_list = init_params.pop('transforms')
        for trans_cfg in trans_list:
            r_trans.append(get_trans(trans_cfg))
        return transforms.RandomApply(r_trans, **init_params)

    elif hasattr(transforms, type_name):
        return getattr(transforms, type_name)(**init_params)
    else:
        raise NotImplementedError(
            "Transform {} is unimplemented".format(trans_cfg))


class BaseTransform(object):
    """ For torch transform or self write
    """
    def __init__(self, pipeline):
        """ transforms for data

        Args:
            pipelines (list): list of dict, each dict is a transform
        """
        self.pipeline = pipeline
        self.transform = self.init_trans(pipeline)

    def init_trans(self, trans_list):
        trans_funcs = []
        for trans_cfg in trans_list:
            trans_funcs.append(get_trans(trans_cfg))
        return transforms.Compose(trans_funcs)

    def __call__(self, data):
        return self.transform(data)


class ListTransform(BaseTransform):
    """ For torch transform or self write
    """
    def __init__(self, pipelines):
        """ transforms for data

        Args:
            pipelines (list): list of dict, each dict is a transform
        """
        self.pipelines = pipelines
        self.transforms = []
        for trans_dict in self.pipelines:
            self.transforms.append(self.init_trans(trans_dict))

    def __call__(self, data):
        results = []
        for trans in self.transforms:
            results.append(trans(data))
        return results
