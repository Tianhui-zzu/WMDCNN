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

data_path = '../public_data/128'


def get_data(file_name='batch1_rebuild.txt'):
    data = pd.read_csv(os.path.join(data_path, file_name), sep=',').to_numpy()  # 转为numpy下数组ndarray形式
    print(data.shape)
    data_y = data[:, 0]  # 第一列是类别,分类标签,减1,label要从1开始
    data_y2 = data[:, 1]  # 第二列是浓度,回归标签
    data_x = data[:, 2:]  # 样本数据
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def get_trainData(batch_size=20, data_x=None, data_y=None):
    train_x = torch.as_tensor(data_x, dtype=torch.float32)
    train_y = torch.as_tensor(data_y, dtype=int)
    train_data = Data.TensorDataset(train_x, train_y)
    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def get_testData(batch_size=20, data_x=None, data_y=None):
    test_x = torch.as_tensor(data_x, dtype=torch.float32)
    test_y = torch.as_tensor(data_y, dtype=int)
    test_data = Data.TensorDataset(test_x, test_y)
    test_dataloader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return test_dataloader


def normalize(x):
    # print('归一化之前:',x)
    x_mean = torch.mean(x, dim=1, keepdim=True)
    # print('x_mean',x_mean)
    x_std = torch.std(x, dim=1, keepdim=True)
    # print('x_std',x_std)
    # # 数据标准化
    x1 = (x - x_mean) / x_std
    return x1


def accuracy(output, labels):
    predictions = torch.max(output, 1)[1]
    labels = labels.data.view_as(predictions)
    right_num = predictions.eq(labels).sum()
    return right_num / len(labels)


def predict(model, device, test_loader):
    all_accuracy = 0
    batch_num = 0
    for batch_size, (test_x, test_y) in enumerate(test_loader):
        # 输入第一层1d卷积要将数据重塑为(batch_size,channel,size1)
        test_x = test_x.view(test_x.shape[0], test_x.shape[2], test_x.shape[1])
        test_x = test_x.to(device)
        labels = test_y.to(device)
        output = model.forward(test_x)
        # 直接接收预测准确率
        all_accuracy = all_accuracy + accuracy(output, labels)
        batch_num = batch_num + 1
    return round((all_accuracy.detach().cpu().numpy() / batch_num), 4)


def draw(df1, df2):
    x1 = np.arange(0, df1.shape[0])
    x2 = np.arange(0, df2.shape[0])
    plt.plot(x1, df1['loss'], linestyle='-', label='value of loss')
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
    plt.plot(x1, df1['train acr'], linestyle='-', label='value of acr', color='red')
    plt.show()
    plt.xlabel('batch num')
    plt.ylabel('value of acr')
    plt.title('value of acr on tested data')
    plt.plot(x1, df1['test acr'], linestyle='-', label='value of acr', color='red')
    plt.show()
    plt.xlabel('step num')
    plt.ylabel('value of learning_rate')
    plt.title('value of learning_rate')
    plt.plot(x2, df2['learning_rate'], linestyle='-', label='value of acr', color='red')
    plt.show()


def adjust_learning_rate(optimizer, epoch, init_lr=0.0001):
    """
    optimizer: 优化器
    epoch: 训练轮数，也可以根据需要加入其它参数
    init_lr：初始学习率，也可以设置为全局变量
    """
    lr = init_lr
    if init_lr > 0.0001:
        lr = init_lr * (0.1 ** (epoch // 150))  # 每35轮下降一次学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    label = np.array(y_train)
    data = np.array(X_train)
    print(data)
    print(label.shape)
    print(data.shape)
