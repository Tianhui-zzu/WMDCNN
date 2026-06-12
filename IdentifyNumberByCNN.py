'卷积神经网络预测数字,运用gpu训练模型,将模型放入device,将训练样本数据放入device'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

input_size = 28  # 图像尺寸为 28*28*1
num_class = 10  # 标签的种类数
num_epochs = 3  # 训练的总循环数
batch_size = 10  # 一个批次的10张照片


# 构建数据集
def get_data():
    # 路径名
    path = 'WMDCNN/data'
    # 训练集
    train_dataset = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(), download=True)
    # 测试集
    test_dataset = datasets.MNIST(root=path, train=False, transform=transforms.ToTensor(), download=True)
    # 按batch划分数据
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


# 打印dataloader,查看其内部格式
def print_dataloader(dataloader):
    for i_batch, batch_data in enumerate(dataloader):
        print(i_batch)
        print(batch_data[0])
        print('batch_data[0].size=', batch_data[0].size())
        print(batch_data[1])
        break  # 打印一轮就结束


# 计算准确率
def accuracy(output, labels):
    predictions = torch.max(output, 1)[1]
    labels = labels.data.view_as(predictions)  # 从中
    rights = predictions.eq(labels).sum()
    # print(rights)
    return rights, len(labels)


# 构建网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层加激活加池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入的通道数,本次训练是灰度图,大小为1*28*28,所以该参数为1
                out_channels=16,  # 输出的通道数,等价于卷积核的个数.
                kernel_size=5,  # 卷积核的尺寸为5*5(2d)
                stride=1,  # 步长为1
                padding=2,  # 填充行数为2,由公式 padding=(kernel_size-1)/2(if padding=1),可保证经过卷积后图像特征图尺寸与原来保持一致,即仍为28*28
            ),  # 卷积后输出(16,28,28)的特征图
            nn.ReLU(),  # relu激活函数
            nn.MaxPool2d(kernel_size=2)  # 池化操作,池化区域为(2*2),每2*2个区域内的数据取最大值代表,经过池化后的结果为:(16,14,14)
        )
        # 第二个卷积层加激活加池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # 卷积后输出为(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化后输出为(32,7,7)
        )
        # 全连接层
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        # print('self.conv1(x).shape:', x.shape)
        # 转换为tensor格式
        x = self.conv2(x) # 此时输出的tensor为：(batch_size,32,7,7) , 4维
        x = x.view(x.size(0), -1)  # view方法类似于reshape方法,将tensor维度重新规划两维,(batch_size,32*7*7)
        return self.out(x)  # 得到结果




# 创建模型
model = CNN()
# !!!!!! 将模型放入gpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# 生成损失函数
loss_func = nn.CrossEntropyLoss()
# 生成优化器
optim = optim.Adam(model.parameters(), lr=0.01)

train_dataloader, test_dataloader = get_data()
# print_dataloader(train_dataloader)
# 传入参数分别是,训练轮次,训练集,验证集,优化器,损失函数,模型
def train_model(num_epochs, train_dataloader, test_dataloader, optim, loss_func,model):
    i = 1
    for epoch in range(num_epochs):
        train_rights = []  # 用于存储一次量为batch的训练下,训练的数据中正确的个数加上训练总数(即batch的大小)
        # 训练过程
        for batch_index, (train_x, train_y) in enumerate(train_dataloader):
            # train_x获取训练的数据,train_y获取对应数据的标签
            model.train()  # 训练过程
            # !!!!将训练数据放入gpu中
            # if i==1:
            #     print('train_x.shape',train_x.shape)
            # i=i+1
            input_data = train_x.to(device)
            label = train_y.to(device)
            # 一次前向传播
            output_data = model.forward(input_data)
            # print(output_data)
            loss = loss_func(output_data, label)  # 计算损失
            # print(f'当前批次{batch_index+1}:loss为{loss}')
            # 反向传播
            loss.backward()  # 反向传播
            optim.step()  # 更新参数
            optim.zero_grad()  # 梯度清零
            # 调用accuracy函数,获取一个元组tup_train=(rights, len(labels))
            tup_train = accuracy(output_data, label)
            train_rights.append(tup_train)
            # 验证过程,每一百个batch验证一次训练模型
            if batch_index % 100 == 0:
                model.eval()  # 验证过程
                val_rights = []  # 用于记录验证集的训练正确的图像个数,以及训练的图像总数
                for batch_index,(test_x, test_y) in enumerate(test_dataloader):
                    # !!!!将训练数据放入gpu中
                    input_data = test_x.to(device)
                    label = test_y.to(device)
                    # 只需要前向传播计算结果，并调用accuracy函数获取元组tup_test
                    output_data = model.forward(input_data)
                    tup_test = accuracy(output_data, label)
                    val_rights.append(tup_test)
                #  准确率计算
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                test_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
                print('训练集正确率:{:.2f},验证集正确率:{:.2f}'.format((train_r[0].cpu().numpy() / train_r[1]),
                                                           (test_r[0].cpu().numpy() / test_r[1])))

train_model(num_epochs, train_dataloader, test_dataloader, optim, loss_func,model)
