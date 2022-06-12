import logging
import math
from cv2 import split

import numpy as np
from PIL import Image
from torchvision import datasets, transforms

import medmnist
from medmnist import INFO, Evaluator

from .transforms.randaugment import RandAugmentMC

# split CIFAR10 into labled/unlabed/val set
def get_pathmnist(args, root):
    transform_labeled = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    base_dataset = medmnist.dataset.PathMNIST(split='train', download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

  def x_u_split(args, labels):
    if isinstance(args.num_labeled, str) and args.num_labeled == "sup":
        return list(range(len(labels))), list(range(len(labels)))

    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.get("expand_labels", False) or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx
  class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 anno_file=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        # the highest prioirty for loading data from file
        if anno_file is not None:
            logging.info("Loading from file {}".format(anno_file))
            anno_data = np.load(anno_file, allow_pickle=True).item()
            self.data = np.array(anno_data["data"])
            self.targets = np.array(anno_data["label"])
            if len(self.data) < 64:
                repeat_num = 64//len(self.data) + 1
                self.data = np.tile(self.data, [repeat_num, 1, 1, 1])
                self.targets = np.tile(self.targets, repeat_num)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
      DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
      
