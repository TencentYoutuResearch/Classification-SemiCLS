""" The Code is under Tencent Youtu Public Rule
"""
import json
import os
import random

import torch
from torchvision import datasets
from torchvision.datasets.folder import default_loader

# prepare ImageNet for semi-superised learning
class ImageNet(datasets.VisionDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.
    """  # noqa: E501
    def __init__(self,
                 root,
                 split,
                 loader=default_loader,
                 transform=None,
                 target_transform=None) -> None:
        super(ImageNet, self).__init__(root,
                                       transform=transform,
                                       target_transform=target_transform)
        self.root = root
        self.split = split
        samples = self.load_annotations()
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.loader = loader

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def load_annotations(self):
        with open(os.path.join(self.root, "{}.txt".format(self.split)),
                  "r") as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        samples = [[x[0], int(x[1])] for x in samples]
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


def get_imagenet_ssl_dataset(root, percent, anno_file, transform_labeled,
                             transform_ulabeled, transform_val):
    """
    Args:
        root: datapath for your imagenet
        precent: the ratio of labeled data to total data
        transform_labeled: transforms pipeline for labeled datatset
        transform_unlabeled: transforms pipeline for unlabeled datatset
        transform_val: transforms pipeline for test datatset

    Returns:
        tuple: (labeled_dataset, unlabeled_dataset, test_dataset)
    """
    labeled_dataset = ImageNet(root=root,
                               split='train',
                               transform=transform_labeled)
    unlabeled_dataset = ImageNet(root=root,
                                 split='train',
                                 transform=transform_ulabeled)

    test_dataset = ImageNet(root=root, split='val', transform=transform_val)

    if not os.path.exists(anno_file):
        # randomly sample labeled data on main process (gpu0)
        if percent == 1:
            label_per_class = 13
        elif percent == 10:
            label_per_class = 128
        else:
            raise ValueError("Unsupported percent {}".format(percent))
        random.shuffle(labeled_dataset.samples)
        labeled_samples = []
        unlabeled_samples = []
        num_img = torch.zeros(1000)
        for i, (img, label) in enumerate(labeled_dataset.samples):
            if num_img[label] < label_per_class:
                labeled_samples.append((img, label))
                num_img[label] += 1
            else:
                unlabeled_samples.append((img, label))
        annotation = {
            'labeled_samples': labeled_samples,
            'unlabeled_samples': unlabeled_samples
        }
        with open(anno_file, 'w') as f:
            json.dump(annotation, f)
        print('save annotation to %s' % anno_file)

    print('load annotation from %s' % anno_file)
    annotation = json.load(open(anno_file, 'r'))

    if percent == 1:
        # repeat labeled samples for faster dataloading
        labeled_dataset.samples = annotation['labeled_samples'] * 10
    else:
        labeled_dataset.samples = annotation['labeled_samples']
    unlabeled_dataset.samples = annotation['unlabeled_samples']

    return labeled_dataset, unlabeled_dataset, test_dataset
