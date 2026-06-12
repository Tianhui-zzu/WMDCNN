'利用pca提取特征，可视化handled_data1_1hz的数据的分布'
import os.path

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['font.family'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

file_name_02 = 'extract_data/batch1+0.9144/extract_y.csv'
file_name_01 = 'extract_data/batch1+0.9144/extract_x_f.csv'


def get_data01(path_x, path_y):
    extract_x = pd.read_csv(path_x)
    extract_y = pd.read_csv(path_y)
    print(extract_x.isnull().any())
    extract_x = extract_x.to_numpy()
    extract_y = extract_y.to_numpy()
    extract_y = extract_y.reshape(extract_y.shape[0])
    return extract_x, extract_y


def vision_data(file_name_01=None, file_name_02=None):
    # ax = Axes3D(plt.figure())
    X, Y = get_data01(file_name_01, file_name_02)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(X_pca)
    # 可视化结果
    # 1: Ethanol; 2: Ethylene; 3: Ammonia; 4: Acetaldehyde; 5: Acetone; 6: Toluene
    colors = ['navy', 'turquoise', 'darkorange', 'red', 'green', 'blue']
    labels = ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene']
    nums = [0, 1, 2, 3, 4, 5]
    markers = ['o', 'o', 'o', 'o', 'o', 'o']
    for color, label, num, marker in zip(colors, labels, nums, markers):
        plt.scatter(X_pca[Y == num, 0], X_pca[Y == num, 1], marker=marker, alpha=.8, label=label,
                    color=color)
    plt.legend(loc='upper left', shadow=False, ncol=2)
    # plt.title(path)
    plt.title("Batch1"+file_name_01.split('/')[1])
    plt.xlabel('PCA1', fontdict={'size': 8}, labelpad=5)
    plt.ylabel('PCA2', fontdict={'size': 8}, labelpad=5)
    plt.show()


# vision_data()
if __name__ == '__main__':
    # data, label = get_data01()
    # data, label = get_data01(os.path.join(path, file_name_01), os.path.join(path, file_name_02))
    # print(label.shape)
    # print(data.shape)
    vision_data(file_name_01='extract_data/batch1_rebuild.txt+0.9212/extract_x_f.csv', file_name_02='extract_data/batch1_rebuild.txt+0.9212/extract_y.csv')
