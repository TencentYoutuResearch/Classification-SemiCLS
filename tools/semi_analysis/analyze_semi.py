""" The Code is under Tencent Youtu Public Rule
analyze simi
"""
import os
from collections import defaultdict

import fire
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn import manifold
from sklearn.metrics.pairwise import cosine_similarity


def _get_feature_list(dir_name):
    """ read data to get dict with key as cls vaulue as feature
    """
    files = [x for x in os.listdir(dir_name) if x.endswith(".npy")]

    # collect feature
    feature_list = defaultdict(list)
    for file_name in files:
        file_path = os.path.join(dir_name, file_name)
        file_data = np.load(file_path, allow_pickle=True).item()
        outputs = file_data['outputs']
        # softmax = file_data['softmax']
        targets = file_data['targets']
        for idx in range(len(targets)):
            feature_list[targets[idx]].append(outputs[idx])
    return feature_list


def write_similarity_cls_graph(dir_name, output_path="./semi_cifar10_cls_similarity.jpg"):
    feature_list = _get_feature_list(dir_name)
    print("classes num: ", len(feature_list.keys()))
    # average feature
    for key in feature_list.keys():
        feature_list[key] = np.mean(feature_list[key], axis=0)

    # calculate sim
    keys = sorted(list(feature_list.keys()))
    feats = [feature_list[x] for x in keys]
    sim_matrix = cosine_similarity(feats)

    # draw plot
    print(keys)
    print("sim_matrix")
    for idx in range(sim_matrix.shape[0]):
        print(" ".join([str(x) for x in sim_matrix[idx]]))

    x = np.repeat(keys, 10)
    y = keys * 10
    z = sim_matrix.reshape(-1)
    fig = go.Figure(
        data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
    fig.write_html(output_path)
    fig.show()


def write_tsne_graph(dir_name, output_path="tsne_graph.jpg"):
    feature_list = _get_feature_list(dir_name)
    '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)

    feats_x, feats_y = [], []
    for key, value in feature_list.items():
        feats_x.extend(value)
        feats_y.extend([key] * len(value))
    feats_x = np.array(feats_x)
    feats_y = np.array(feats_y)

    del feature_list
    print("fitting tsne")
    feats_x_tsne = tsne.fit_transform(feats_x)
    print("Org data dimension is {}. Embedded data dimension is {}".format(
        feats_x.shape[-1], feats_x_tsne.shape[-1]))
    '''嵌入空间可视化'''
    x_min, x_max = feats_x_tsne.min(0), feats_x_tsne.max(0)
    X_norm = (feats_x_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0],
                 X_norm[i, 1],
                 str(feats_y[i]),
                 color=plt.cm.Set1(feats_y[i]),
                 fontdict={
                     'weight': 'bold',
                     'size': 9
                 })
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_path)


if __name__ == '__main__':
    fire.Fire()
