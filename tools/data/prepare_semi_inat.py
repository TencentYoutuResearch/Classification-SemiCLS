""" The Code is under Tencent Youtu Public Rule
Generate semi-inat 2021 txt files for semi-inat dataset
Usage:

python3 tools/data/prepare_semi_inat.py /semi-inat-2021/folder/path
"""

import os
import sys

from tqdm import tqdm


def convert_labeled(folder_dir):
    output_folder = os.path.dirname(folder_dir)

    anno_list = []
    for folder_name in tqdm(os.listdir(folder_dir)):
        folder_path = os.path.join(folder_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for jpg_name in os.listdir(folder_path):
            if not jpg_name.endswith(".jpg"):
                continue
            jpg_path = os.path.join(folder_path, jpg_name)
            anno_list.append([jpg_path, int(folder_name)])

    anno_path = os.path.join(output_folder, "anno.txt")
    print("writing labeled anno to ", anno_path)
    with open(anno_path, "w") as f:
        for line in anno_list:
            f.write("{} {}\n".format(line[0], line[1]))


def convert_ulabeled(folder_dir):
    output_folder = os.path.dirname(folder_dir)

    anno_list = []
    for jpg_name in tqdm(os.listdir(folder_dir)):
        if not jpg_name.endswith(".jpg"):
            continue
        jpg_path = os.path.join(folder_dir, jpg_name)
        anno_list.append(jpg_path)

    anno_path = os.path.join(output_folder, "u_train.txt")
    print("writing unlabeled anno to ", anno_path)
    with open(anno_path, "w") as f:
        for line in anno_list:
            f.write("{}\n".format(line))

if __name__ == '__main__':

    semi_inat_dir = os.path.realpath(sys.argv[1])
    l_train_dir = os.path.join(semi_inat_dir, "l_train")
    u_train_dir = os.path.join(semi_inat_dir, "u_train")
    val_dir = os.path.join(semi_inat_dir, "val")

    convert_ulabeled(os.path.join(u_train_dir, "u_train"))
    convert_labeled(os.path.join(l_train_dir, "l_train"))
    convert_labeled(os.path.join(val_dir, "val"))


