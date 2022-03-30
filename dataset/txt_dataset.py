""" The Code is under Tencent Youtu Public Rule
"""
import json
import os
import random

import torch
from torchvision import datasets
from torchvision.datasets.folder import default_loader


class TxtDataset(datasets.VisionDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.
    """  # noqa: E501
    def __init__(self,
                 anno_file,
                 loader=default_loader,
                 transform=None,
                 target_transform=None) -> None:
        super(TxtDataset, self).__init__(anno_file,
                                         transform=transform,
                                         target_transform=target_transform)
        self.anno_file = anno_file
        samples = self.load_annotations()
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.loader = loader

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def load_annotations(self):
        with open(self.anno_file, "r") as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        if len(samples[0]) == 2:
            samples = [[x[0], int(x[1])] for x in samples]
        elif len(samples[0]) == 1:
            samples = [[x[0], 0] for x in samples]

        return samples

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def get_txt_ssl_dataset(l_anno_file, u_anno_file, v_anno_file,
                        transform_labeled, transform_ulabeled, transform_val):
    """
    Args:
        *_anno_file (): annotation file path

        transform_*: transform pipeline for labeled/unlabeled/test set.

    Returns:
        tuple: (labeled_dataset, unlabeled_dataset, test_dataset)
    """
    labeled_dataset = TxtDataset(anno_file=l_anno_file,
                                 transform=transform_labeled)
    unlabeled_dataset = TxtDataset(anno_file=u_anno_file,
                                   transform=transform_ulabeled)

    test_dataset = TxtDataset(anno_file=v_anno_file, transform=transform_val)
    return labeled_dataset, unlabeled_dataset, test_dataset
