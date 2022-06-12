""" The Code is under Tencent Youtu Public Rule
This file build a dataset for semi supervised learning
dataset_cfg:
    num_labeled:
    num_classes:
    batch_size:
    eval_step:

    root:
    type: cifar10
    lpipelines:
    upipelinse:
    vpipeline:
"""
from torchvision import datasets

from dataset.cifar import CIFAR10SSL, CIFAR100SSL, x_u_split
from dataset.pathmnist import get_pathmnist
from dataset.imagenet import get_imagenet_ssl_dataset
from dataset.MyDataset import MyDataset
from dataset.stl10 import get_stl10
from dataset.transforms.builder import BaseTransform, ListTransform
from dataset.txt_dataset import get_txt_ssl_dataset

dataset_dict = {"CIFAR10SSL": CIFAR10SSL, "CIFAR100SSL": CIFAR100SSL}

base_dict = {
    "CIFAR10": datasets.CIFAR10,
    "CIFAR10SSL": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
    "CIFAR100SSL": datasets.CIFAR100,
    "MyDataset": MyDataset,
}


def build(cfg):

    if cfg.type == "MyDataset":
        transform_ulabeled = ListTransform(cfg.upipelinse)
        train_unlabeled_dataset = datasets.ImageFolder(
            root=cfg.root,
            transform=transform_ulabeled)
        transform_labeled = ListTransform(cfg.lpipelines)
        train_labeled_dataset = MyDataset(names_file=cfg.labeled_names_file, transform=transform_labeled)
        transform_val = BaseTransform(cfg.vpipeline)
        test_dataset = MyDataset(names_file=cfg.test_names_file, transform=transform_val)

    elif cfg.type == "ImagenetSSL":
        transform_labeled = ListTransform(cfg.lpipelines)
        transform_ulabeled = ListTransform(cfg.upipelinse)
        transform_val = BaseTransform(cfg.vpipeline)
        return get_imagenet_ssl_dataset(root=cfg.root,
                                        percent=cfg.percent,
                                        anno_file=cfg.anno_file,
                                        transform_labeled=transform_labeled,
                                        transform_ulabeled=transform_ulabeled,
                                        transform_val=transform_val)

    elif cfg.type == "TxtDatasetSSL":
        transform_labeled = ListTransform(cfg.lpipelines)
        transform_ulabeled = ListTransform(cfg.upipelinse)
        transform_val = BaseTransform(cfg.vpipeline)
        return get_txt_ssl_dataset(
            l_anno_file=cfg.l_anno_file,
            u_anno_file=cfg.u_anno_file,
            v_anno_file=cfg.v_anno_file,
            transform_labeled=transform_labeled,
            transform_ulabeled=transform_ulabeled,
            transform_val=transform_val)

    elif cfg.type == "STL10SSL":
        transform_labeled = ListTransform(cfg.lpipelines)
        transform_ulabeled = ListTransform(cfg.upipelinse)
        transform_val = BaseTransform(cfg.vpipeline)
        return get_stl10(
            root=cfg.root, folds=cfg.folds,
            transform_labeled=transform_labeled,
            transform_ulabeled=transform_ulabeled,
            transform_val=transform_val)
    
    elif cfg.type == "PATHMNIST":
        return get_pathmnist(cfg)

    else:

        base_dataset = base_dict[cfg.type](cfg.root, train=True, download=True)

        train_labeled_idxs, train_unlabeled_idxs = x_u_split(
            cfg, base_dataset.targets)

        # init labeled datasets
        transform_labeled = ListTransform(cfg.lpipelines)
        train_labeled_dataset = dataset_dict[cfg.type](
            cfg.root,
            train_labeled_idxs,
            train=True,
            transform=transform_labeled,
            anno_file=cfg.lanno_file if cfg.get("lanno_file", False) else None)

        transform_ulabeled = ListTransform(cfg.upipelinse)
        train_unlabeled_dataset = dataset_dict[cfg.type](
            cfg.root,
            train_unlabeled_idxs,
            train=True,
            transform=transform_ulabeled,
            anno_file=cfg.uanno_file if cfg.get("uanno_file", False) else None)

        transform_val = BaseTransform(cfg.vpipeline)
        test_dataset = base_dict[cfg.type](cfg.root,
                                        train=False,
                                        transform=transform_val,
                                        download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
