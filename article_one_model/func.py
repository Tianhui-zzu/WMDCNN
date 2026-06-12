import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from train import train

# import hiddenlayer as h1
'用自搭建卷积神经网络检测气体类别'


path = 'extract_feature_vectors+area+mean+max+max+std.csv'

def get_data():
    data = pd.read_csv(path, sep=',',names=[i for i in range(0,41)]).to_numpy()  # 转为numpy下数组ndarray形式
    print('train_data.shape:', data.shape)
    data_y = data[:, 0]  # 第一列是类别,分类标签,减1,label要从1开始
    data_x = data[:, 1:]  # 样本数据

    # ls_data = np.array(ls_data)
    # ls_label = np.array(ls_label)
    # print('ls_data', ls_data)
    # print('ls_label', ls_label)
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.4, random_state=42)

    return X_train, X_test, y_train, y_test


def get_trainData(batch_size=20, data_x=None, data_y=None):
    train_x = torch.as_tensor(data_x, dtype=torch.float32)
    train_y = torch.as_tensor(data_y, dtype=torch.int64)
    train_data = Data.TensorDataset(train_x, train_y)
    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    return train_dataloader


def get_testData(batch_size=20, data_x=None, data_y=None):
    test_x = torch.as_tensor(data_x, dtype=torch.float32)
    test_y = torch.as_tensor(data_y, dtype=torch.float32)
    test_data = Data.TensorDataset(test_x, test_y)
    test_dataloader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return test_dataloader




def draw(df):
    x = np.arange(0, df.shape[0])
    plt.plot(x, df['loss'], linestyle='-', label='value of loss')
    # plt.plot(x,df['test acr'],linestyle='-',label='value of loss')
    # plt.plot(x,df['train acr'],linestyle='-',label='value of loss')
    plt.legend(loc='upper right')
    plt.xlabel('batch num')
    plt.ylabel('value of loss')
    plt.title('value of loss')
    plt.show()
    plt.xlabel('batch num')
    plt.ylabel('value of acr')
    plt.title('value of acr on trained data')
    plt.plot(x, df['train acr'], linestyle='-', label='value of acr', color='red')
    plt.show()
    plt.xlabel('batch num')
    plt.ylabel('value of acr')
    plt.title('value of acr on tested data')
    plt.plot(x, df['test acr'], linestyle='-', label='value of acr', color='red')
    plt.show()


