import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
from Train import train

# import hiddenlayer as h1
'用自搭建卷积神经网络检测气体类别'
epochs = 15
G = 2  # 要预测的气体的类别个数
batch_size = 5
predict_time = 4
lr = 0.001
print_batch = 5
# os.environ["CUDA_VISIBLE_DEVICES"]='cuda:1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'用batch2_rebuild作为验证集,batch3_rebuild作为训练集'


def get_trainData():
    path = '../WMDCNN/data/my_data/硫化氢(前80)加氨气(后30).txt'
    data = pd.read_csv(path, sep=',').to_numpy()  # 转为numpy下数组ndarray形式
    print(data.shape)
    data_y1 = data[:, 0]  # 第一列是类别,分类标签
    data_y2 = data[:, 2]  # 第三列是浓度,回归标签
    data_x = data[:, 3:]  # 样本数据
    train_x = torch.from_numpy(data_x.astype(np.float32))
    train_y1 = torch.from_numpy(data_y1.astype(np.int64))
    train_y2 = torch.from_numpy(data_y2.astype(np.float32))
    train_data1 = Data.TensorDataset(train_x, train_y1)
    train_data2 = Data.TensorDataset(train_x, train_y2)
    train_dataloader1 = Data.DataLoader(dataset=train_data1, batch_size=batch_size, shuffle=True, num_workers=1)
    train_dataloader2 = Data.DataLoader(dataset=train_data2, batch_size=batch_size, shuffle=True, num_workers=1)
    # for step, (bx, by) in enumerate(train_dataloader1):
    #     print('bx.shape', bx.shape)
    #     print('bx', bx)
    #     print('by.shape', by.shape)
    #     print('by', by)
    #     print('bx.dtype', bx.dtype)
    #     print('by.dtype', by.dtype)
    #     break
    train_dataloader = [train_dataloader1, train_dataloader2]
    return train_dataloader


def get_testData():
    path = '../WMDCNN/data/my_data/硫化氢(前80)加氨气(后30).txt'  # 用batch2当作训练集,batch3当作测试集
    data = pd.read_csv(path, sep=',').to_numpy()  # 转为numpy下数组ndarray形式
    print(data.shape)
    data_y1 = data[:, 0]  # 第一列是类别,分类标签
    data_y2 = data[:, 2]  # 第三列是浓度,回归标签
    data_x = data[:, 3:]  # 样本数据
    test_x = torch.from_numpy(data_x.astype(np.float32))
    test_y1 = torch.from_numpy(data_y1.astype(np.int64))
    test_y2 = torch.from_numpy(data_y2.astype(np.float32))
    test_data1 = Data.TensorDataset(test_x, test_y1)
    test_data2 = Data.TensorDataset(test_x, test_y2)
    test_dataloader1 = Data.DataLoader(dataset=test_data1, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataloader2 = Data.DataLoader(dataset=test_data2, batch_size=batch_size, shuffle=True, num_workers=1)
    # for step, (bx, by) in enumerate(test_dataloader1):
    #     print('bx.shape', bx.shape)
    #     print('bx', bx)
    #     print('by.shape', by.shape)
    #     print('by', by)
    #     print('bx.dtype', bx.dtype)
    #     print('by.dtype', by.dtype)
    test_dataloader = [test_dataloader1, test_dataloader2]
    return test_dataloader


class H_CNN(nn.Module):
    def __init__(self):
        super(H_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,  #
                out_channels=18,  # 输出的通道数,等价于卷积核的个数.
                kernel_size=3,  # 卷积核的尺寸为1*3(1d)
                stride=1,  # 步长为1
                padding=1,  # 填充行数为1,由公式 padding=(kernel_size-1)/2(if padding=1),可保证经过卷积后图像特征图尺寸与原来保持一致,即仍为28*28
            ),
            nn.MaxPool1d(kernel_size=3),  # 池化操作,池化区域为(2*2),每2*2个区域内的数据取最大值代表,经过池化后的结果为:(16,14,14)
            nn.Dropout(p=0.25)  # 25%概率将值置为0
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=3,  # 卷积核的尺寸为3*3(2d)
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=3),  # 3d卷积
            nn.Dropout(p=0.25)
        )
        self.conv3 = nn.Sequential(
            # nn.Conv3d(
            #     in_channels=1,
            #     out_channels=1,
            #     kernel_size=3,  # 卷积核的尺寸为3*3*3(3d)
            #     stride=1,
            #     padding=1,
            # ),
            # # nn.MaxPool3d(kernel_size=2),
            # nn.Conv3d(
            #     in_channels=1,
            #     out_channels=1,
            #     kernel_size=3,  # 卷积核的尺寸为3*3*3(3d)
            #     stride=1,
            #     padding=1,
            # ),
            # nn.MaxPool3d(kernel_size=2),  # 3d卷积
            nn.Conv3d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,  # 卷积核的尺寸为3*3*3(3d)
                stride=1,
                padding=1,
            ),
            nn.MaxPool3d(kernel_size=3),  # 3d卷积
            nn.Dropout(p=0.25)
        )
        self.out = nn.Linear(3 * 2 * 2 * 2, G)
        # self.out = nn.Linear(10 * G, G)

    def forward(self, x):
        x = self.conv1(x)
        # print(x)
        # print('1d卷积后x.shape', x.shape)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        # print('x.shape',x)
        x = self.conv2(x)
        # print('2d卷积后x.shape', x.shape)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
        x = self.conv3(x)
        # print('3d卷积后x.shape', x.shape)
        x = x.view(x.size(0), -1)  # view方法类似于reshape方法,将tensor维度重新规划两维,(batch_size,32*7*7)
        # x = self.dense(x)
        # print('dense()后x.shape', x.shape)
        x = self.out(x)
        # print('全连接层后x.shape', x.shape)
        # print('全连接层后x', x.shape)
        return x


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


print_batch = 5

if __name__ == '__main__':
    # get_trainData()
    print(torch.__version__)
    model = H_CNN()
    # 生成损失函数
    loss_func = nn.CrossEntropyLoss()
    # 生成优化器
    optim = optim.Adam(model.parameters(), lr=lr)
    test_loader = get_testData()[0]
    train_loader = get_trainData()[0]
    # train_model(optim, loss_func, model, train_loader, test_loader)
    loss_list_epoch = train(model, device, train_loader, test_loader, optim, loss_func, epochs, print_batch)
    draw(pd.DataFrame(loss_list_epoch))
