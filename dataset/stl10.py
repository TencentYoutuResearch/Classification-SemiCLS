""" The Code is under Tencent Youtu Public Rule
"""
from torchvision.datasets.stl10 import STL10


# prepare STL-10 for semi-superised learning
# return tuple(labeled_dataset, unlabeled_dataset, test_dataset)

"""
prepare STL-10 for semi-superised learning
return tuple(labeled_dataset, unlabeled_dataset, test_dataset)
"""
def get_stl10(root, folds, transform_labeled, transform_ulabeled,
              transform_val):

    labeled_dataset = STL10(root,
                            "train",
                            folds=folds,
                            transform=transform_labeled)
    unlabeled_dataset = STL10(root,
                              "unlabeled",
                              folds=folds,
                              transform=transform_ulabeled)
    test_dataset = STL10(root, "test", folds=folds, transform=transform_val)
    return labeled_dataset, unlabeled_dataset, test_dataset
