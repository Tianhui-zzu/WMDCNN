'增强卷积神经网络(ACNN)补偿气体漂移-学习模型'
import math

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os.path as op
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# import hiddenlayer as h1
'用自搭建卷积神经网络检测气体类别'
epochs = 20
# 增强训练的批次
aug_epochs = 20
G = 3  # 要预测的气体的类别个数
batch_size = 10
test_batch_size = 50
predict_time = 6
lr = 0.0001
# os.environ["CUDA_VISIBLE_DEVICES"]='cuda:1'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
'用batch1_rebuild作为验证集,batch2_rebuild作为训练集'


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 显示负号


# 构建数据集
def get_trainData():
    path = '../public_data/gas Sensor Array Drift Dataset at Different Concentrations/b1.txt'
    data = pd.read_csv(path, sep=',').to_numpy()  # 转为numpy下数组ndarray形式
    print(data.shape)
    data_y1 = data[:, 0]  # 第一列是类别,分类标签,减1,label要从1开始
    data_y2 = data[:, 1]  # 第二列是浓度,回归标签
    data_x = data[:, 2:]  # 样本数据
    train_x = torch.from_numpy(data_x.astype(np.float32))
    train_y1 = torch.from_numpy(data_y1.astype(np.int64))
    train_y2 = torch.from_numpy(data_y2.astype(np.float32))
    train_data1 = Data.TensorDataset(train_x, train_y1)
    train_data2 = Data.TensorDataset(train_x, train_y2)
    train_dataloader1 = Data.DataLoader(dataset=train_data1, batch_size=batch_size, shuffle=True, num_workers=1)
    train_dataloader2 = Data.DataLoader(dataset=train_data2, batch_size=batch_size, shuffle=True, num_workers=1)
    # for step, (bx, by) in enumerate(train_dataloader1):
    #     # print('bx.shape', bx.shape)
    #     # print('bx', bx)
    #     # print('by.shape', by.shape)
    #     # print('by', by)
    #     # print('bx.dtype', bx.dtype)
    #     # print('by.dtype', by.dtype)
    #     # break
    train_dataloader = [train_dataloader1, train_dataloader2]
    return train_dataloader


def get_testData():
    path = '../public_data/gas Sensor Array Drift Dataset at Different Concentrations/b2.txt'  # 用batch2当作训练集,batch3当作测试集
    data = pd.read_csv(path, sep=',').to_numpy()  # 转为numpy下数组ndarray形式
    print(data.shape)
    data_y1 = data[:, 0]  # 第一列是类别,分类标签
    data_y2 = data[:, 1]  # 第二列是浓度,回归标签
    data_x = data[:, 2:]  # 样本数据
    test_x = torch.from_numpy(data_x.astype(np.float32))
    test_y1 = torch.from_numpy(data_y1.astype(np.int64))
    test_y2 = torch.from_numpy(data_y2.astype(np.float32))
    test_data1 = Data.TensorDataset(test_x, test_y1)
    test_data2 = Data.TensorDataset(test_x, test_y2)
    test_dataloader1 = Data.DataLoader(dataset=test_data1, batch_size=test_batch_size, shuffle=True, num_workers=1)
    test_dataloader2 = Data.DataLoader(dataset=test_data2, batch_size=test_batch_size, shuffle=True, num_workers=1)
    # for step, (bx, by) in enumerate(test_dataloader1):
    #     print('bx.shape', bx.shape)
    #     print('bx', bx)
    #     print('by.shape', by.shape)
    #     print('by', by)
    #     print('bx.dtype', bx.dtype)
    #     print('by.dtype', by.dtype)
    #     break
    test_dataloader = [test_dataloader1, test_dataloader2]
    return test_dataloader


def get_validateData():
    path = '../WMDCNN/data/128/b2.txt'  # 用batch2当作训练集,batch3当作测试集
    data = pd.read_csv(path, sep=',').to_numpy()  # 转为numpy下数组ndarray形式
    print(data.shape)
    data_y1 = data[:, 0]  # 第一列是类别,分类标签
    data_y2 = data[:, 1]  # 第二列是浓度,回归标签
    data_x = data[:, 2:]  # 样本数据
    validate_x = torch.from_numpy(data_x.astype(np.float32))
    validate_y1 = torch.from_numpy(data_y1.astype(np.int64))
    validate_y2 = torch.from_numpy(data_y2.astype(np.float32))
    validate_data1 = Data.TensorDataset(validate_x, validate_y1)
    validate_data2 = Data.TensorDataset(validate_x, validate_y2)
    validate_dataloader1 = Data.DataLoader(dataset=validate_data1, batch_size=batch_size, shuffle=False, num_workers=1)
    validate_dataloader2 = Data.DataLoader(dataset=validate_data2, batch_size=batch_size, shuffle=True, num_workers=1)
    # for step, (bx, by) in enumerate(test_dataloader1):
    #     print('bx.shape', bx.shape)
    #     print('bx', bx)
    #     print('by.shape', by.shape)
    #     print('by', by)
    #     print('bx.dtype', bx.dtype)
    #     print('by.dtype', by.dtype)
    #     break
    validate_dataloader = [validate_dataloader1, validate_dataloader2]
    return validate_dataloader


# 打印dataloader,查看其内部格式
def print_dataloader(dataloader):
    for i_batch, batch_data in enumerate(dataloader):
        print(i_batch)
        print(batch_data[0])
        print('batch_data[0].size=', batch_data[0].size())
        print(batch_data[1])
        break  # 打印一轮就结束


def normalize(x):
    # print('归一化之前:',x)
    x_mean = torch.mean(x, dim=1, keepdim=True)
    # print('x_mean',x_mean)
    x_std = torch.std(x, dim=1, keepdim=True)
    # print('x_std',x_std)
    # # 数据标准化
    x1 = (x - x_mean) / x_std
    # print('归一化之后:', x1)
    return x1


# 计算准确率
def accuracy(output, labels):
    predictions = torch.max(output, 1)[1]
    labels = labels.data.view_as(predictions)
    right_num = predictions.eq(labels).sum()
    return right_num / len(labels)


# 构建网络模型
class ACNN(nn.Module):
    def __init__(self):
        super(ACNN, self).__init__()
        # 第一个卷积层加激活加池化,输入为16*8的二维矩阵
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入的通道数,本次为16*8的二维矩阵,所以该参数为1
                out_channels=16,  # 输出的通道数,等价于卷积核的个数.
                kernel_size=1,  # 卷积核的尺寸为1*1(2d)
                stride=1,  # 步长为1
                padding=0,
            ),  # 卷积后输出(16,8,16)的特征图
            nn.ReLU(),  # relu激活函数
            nn.MaxPool2d(kernel_size=2)  # 池化操作,池化区域为(2*2),每2*2个区域内的数据取最大值代表,经过池化后的结果为:(8,4,16)
        )
        # 第二个卷积层加激活加池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0),  # 卷积后输出为(8,4,32)
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化后输出为(4,2,32)
        )
        # 全连接层d1
        self.dense = nn.Linear(4 * 2 * 32, 1 * 128)  # 全连接输出为(1,128)
        # 全连接层d2
        self.out = nn.Linear(1 * 128, 1 * G)  # 1*6最后全连接输出向量长度6代表分类气体个数

    def forward(self, x):
        x = self.conv1(x)
        # print('self.conv1(x).shape:', x.shape)
        # 转换为tensor格式
        x = self.conv2(x)  # 此时输出的tensor为：(batch_size,4,2,32) , 4维
        x = x.view(x.size(0), -1)  # view方法类似于reshape方法,将tensor维度重新规划两维,(batch_size,4*2*32)
        x = self.dense(x)  # 得到结果
        x = self.out(x)
        return x


def predict(model, test_loader):
    all_accuracy = 0
    batch_num = 0
    for batch_index, (test_x, test_y) in enumerate(test_loader):
        # 输入第一层1d卷积要将数据重塑为(batch_size,channel,size1)
        test_x = test_x.reshape(test_x.shape[0], 1, 16, 8)  # 相当于原论文中的transform将128->16*8
        test_x = test_x.to(device)
        labels = test_y.to(device)
        output = model.forward(test_x)
        # 直接接收预测准确率
        all_accuracy = all_accuracy + accuracy(output, labels)
        batch_num = batch_num + 1
        return round((all_accuracy.detach().cpu().numpy() / batch_num), 4)


def draw(df):
    x = np.arange(0, df.shape[0])
    # 绘制第一张图,损失值
    plt.plot(x, df['loss'], linestyle='-', label='value of loss')
    plt.legend(loc='upper right')
    plt.xlabel('batch num')
    plt.ylabel('value of loss')
    plt.title('Simple Model:value of loss on trained data')
    plt.show()
    # 绘制第二张图,预测准确率
    plt.plot(x, df['test acr'], linestyle='-', label='value of acr', color='red')
    # 添加平均线
    average = np.mean(df['test acr'])  # 利用numpy获取成绩的平均数
    # y设定在图中的纵坐标,linestyle设置风格字符,可选有:'_',':'
    plt.axhline(y=average, color='b', linestyle=':')
    plt.legend(loc='upper right')
    plt.xlabel('batch num')
    plt.ylabel('value of acr')
    plt.title('Simple Model:value of acr on trained data')
    plt.show()


def draw_aug(df):
    x = np.arange(0, df.shape[0])
    # 绘制第一张图,损失值
    plt.plot(x, df['loss'], linestyle='-', label='value of loss')
    plt.legend(loc='upper right')
    plt.xlabel('batch num')
    plt.ylabel('value of loss ')
    plt.title('Augmented Model:value of loss on trained data ')
    plt.show()
    # 绘制第二张图,预测准确率
    plt.plot(x, df['test acr'] , linestyle='-', label='value of acr', color='red')
    # 添加平均线
    average = np.mean(df['test acr'])  # 利用numpy获取成绩的平均数
    # y设定在图中的纵坐标,linestyle设置风格字符,可选有:'_',':'
    plt.axhline(y=average, color='b', linestyle=':')

    plt.legend(loc='upper right')
    plt.xlabel('batch num')
    plt.ylabel('value of acr')
    plt.title('Augmented Model:value of acr on trained data')
    plt.show()


'自增强算法涉及参数 w = Xi-Xi+1, epsilon = Ypre-Ytrue, beta = -(epsilon+1)log(1-1/epsilon+1),delta = delta+beta*epsilon'
w = 0
epsilon = 0
beta = 0
delta = 0


# 增强算法
def augument(model, num_epochs, optim, loss_func, train_loader, test_loader, validate_loader):
    loss_acr = {'loss': [], 'test acr': []}
    data1 = pd.read_csv('../WMDCNN/data/128/b1.txt').to_numpy()
    data2 = pd.read_csv('../WMDCNN/data/128/b2.txt').to_numpy()
    # b1,b2是同等数量的数据集，且按顺序将同类别的数据按顺序按同等数量排序，作差，得出偏移矩阵
    W = torch.from_numpy(data1[:, 2:].astype(np.float32)) - torch.from_numpy(data2[:, 2:].astype(np.float32))
    for epoch in range(num_epochs):
        flag = 1
        for batch_index, (validate_x, Ytrue) in enumerate(validate_loader[0]):
            print(f'增强模型中....迭代次数为：{epoch},当前批次:{batch_index + 1}')
            validate_x = normalize(validate_x)
            W = normalize(W)
            validate_x = validate_x.to(device)
            Ytrue = Ytrue.to(device)
            W = W.to(device)
            # 由已经训练完成的模型预测样本,得出类别值
            Ypre = model.forward(validate_x.reshape(validate_x.shape[0], 1, 16, 8))
            # 将输出值batch_size*G按行取最大
            Ypre = torch.max(Ypre, 1)[1]
            # print('Ypre',Ypre)
            # print('Ytrue',Ytrue)
            epsilon = (Ypre - Ytrue)
            # 与beta统一为float类型
            epsilon = torch.tensor(epsilon, dtype=torch.float32)
            # 一维向量->二维矩阵
            epsilon = epsilon.view(epsilon.shape[0], -1)
            # print('epsilon', epsilon)
            beta = -(epsilon + 1) * torch.log(1 - 1 / (epsilon + 3))
            # beta = epsilon
            # print('beta',beta)

            if (flag == 1):
                delta = epsilon.matmul(beta.view(-1, epsilon.shape[0]))
            else:
                delta = delta + epsilon.matmul(beta.view(-1, epsilon.shape[0]))
                flag = flag + 1
            print('delta', delta)
            start = (batch_index) * batch_size
            # 漂移补偿矩阵
            F = delta.matmul(W[start:start + Ytrue.shape[0], :])
            # print('漂移补偿矩阵F', F)
            validate_x = validate_x + F
            # 重新利用新数据集进行模型校正
            validate_x = validate_x.reshape(validate_x.shape[0], 1, 16, 8)
            output = model.forward(validate_x)
            loss = loss_func(output, Ytrue)
            loss_print = round(loss.detach().cpu().numpy() / 1, 4)
            # 打印在训练集上的损失和验证集的预测率
            if batch_index % print_batch == 0:
                loss_acr['loss'].append(loss_print)
                test_acr = predict(model, test_loader[0])
                loss_acr['test acr'].append(test_acr)
                print('validate——loss:', loss_print)
            # 反向传播
            loss.backward()
            optim.step()  # 更新参数
            optim.zero_grad()  # 梯度清零
        df = pd.DataFrame(loss_acr)
    return df


print_batch = 5  # 每过5个批次,打印一次损失和预测值


def train_model(num_epochs, aug_epochs, optim, loss_func, model, train_loader, test_loader, validate_loader):
    model = model.to(device)
    loss_list_epoch = {'index': [], 'loss': [], 'test acr': []}
    '第一步以b1.txt为源数据,训练一个模型model'
    for epoch in range(num_epochs):
        # 记录模型整个训练过程的损失值情况,以epoch为单位
        for batch_index, (train_x, train_y1) in enumerate(train_loader[0]):
            print(f'当前迭代次数：{epoch},当前批次:{batch_index + 1}')
            # print(train_y1.shape[0])
            model.train()
            train_x = normalize(train_x)
            # 输入第一层1d卷积要将数据重塑为(batch_size,channel,size1)
            train_x = train_x.reshape(train_x.shape[0], 1, 16, 8)  # 相当于原论文中的transform将128->16*8
            train_x = train_x.to(device)
            label = train_y1.to(device)
            output = model.forward(train_x)
            # 计算损失值
            print('label:', label)
            loss = loss_func(output, label)
            loss_print = round(loss.detach().cpu().numpy() / 1, 4)
            print('loss:', loss)
            # 反向传播
            loss.backward()
            optim.step()  # 更新参数
            optim.zero_grad()  # 梯度清零
            # 打印在训练集上的损失和验证集的预测率
            if batch_index % print_batch == 0:
                loss_list_epoch['index'].append(str(epoch) + '_' + str(batch_index / print_batch))
                loss_list_epoch['loss'].append(loss_print)
                test_acr = predict(model, test_loader[0])
                loss_list_epoch['test acr'].append(test_acr)

    # print(loss_list_epoch)
    df1 = pd.DataFrame(loss_list_epoch)
    if op.exists(f'data/predict2/predicted_result_{predict_time}.csv'):
        print('当前文件已存在,若要存储新文件,请修改predict_time值')
    else:
        df1.to_csv(f'data/predict2/predicted_result_{predict_time}.csv', index=False, header=None)
        print('训练效果和预测结果保存成功')

    df2 = augument(model, aug_epochs, optim, loss_func, train_loader, test_loader, validate_loader)
    draw(df1)
    draw_aug(df2)
    print('未增强前的模型:', df1)
    print('增强后的模型:', df2)


if __name__ == '__main__':
    print(torch.__version__)
    model = ACNN()
    # model.load_state_dict(torch.load('ACNN_MODEL.pth'))
    # 生成损失函数
    loss_func = nn.CrossEntropyLoss()
    # 生成优化器
    optim = optim.Adam(model.parameters(), lr=lr)
    test_loader = get_testData()
    train_loader = get_trainData()
    validate_loader = get_validateData()
    train_model(epochs, aug_epochs, optim, loss_func, model, train_loader, test_loader, validate_loader)

    # 保存参数文件
    torch.save(model.state_dict(), '../model.pth')

    # test_normalize()
