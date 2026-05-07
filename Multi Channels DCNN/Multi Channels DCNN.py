import netron
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from Self_Attention_Layers import Self_Attention_Layers
from Func import predict, draw, get_trainData, get_testData, normalize, adjust_learning_rate
from tqdm import tqdm
import time

G = 6
lr = 0.001
epochs = 10
predict_time = 4
print_batch = 20
momentum = 0.9
device = torch.device('cuda')


class ConvBlock(nn.Module):

    def __init__(self, channels_1, channels_2, channels_3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels_1, channels_2, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(channels_2, channels_3, kernel_size=3, padding=1, stride=1)
        # 残差 通道数channels_1->channels_3
        self.conv3 = nn.Conv1d(channels_1, channels_3, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        y = F.gelu(self.conv1(x))
        y = self.conv2(y)
        x = self.conv3(x)
        return F.gelu(x + y)


class MC_DCNN(nn.Module):
    def __init__(self):
        super(MC_DCNN, self).__init__()

        # 输出通道数需要调整,原论文没有给出
        # d1,第一平行卷积路径
        self.conv1_d1 = ConvBlock(1, 2, 4)
        self.max_pool1_d1 = nn.MaxPool1d(2)
        self.conv2_d1 = ConvBlock(4, 6, 8)
        self.max_pool2_d1 = nn.MaxPool1d(2)
        self.conv3_d1 = ConvBlock(8, 12, 16)
        # d2,第二平行卷积路径
        self.conv1_d2 = ConvBlock(1, 2, 4)
        self.max_pool1_d2 = nn.MaxPool1d(2)
        self.conv2_d2 = ConvBlock(4, 6, 8)
        self.max_pool2_d2 = nn.MaxPool1d(2)
        self.conv3_d2 = ConvBlock(8, 12, 16)
        # d3,第三平行卷积路径
        self.conv1_d3 = ConvBlock(1, 2, 4)
        self.max_pool1_d3 = nn.MaxPool1d(2)
        self.conv2_d3 = ConvBlock(4, 6, 8)
        self.max_pool2_d3 = nn.MaxPool1d(2)
        self.conv3_d3 = ConvBlock(8, 12, 16)
        #
        self.GAP = nn.AdaptiveAvgPool1d(1)  # !!!!!!!!!!!!!!!!!!!实现全局平均池化作用

        #
        self.self_attention_layers = Self_Attention_Layers()
        # *3代表3个平行卷积路径
        self.out = nn.Linear(16, G)

    def forward(self, x1, x2, x3):
        # Concat
        # x1 = self.conv1(x1)
        # x2 = self.conv2(x2)
        # x2 = self.conv2(x3)

        # x = torch.cat((x1, x2), 1)
        x1 = self.conv1_d1(x1)  # torch.Size([2, 4, 128])       ！！！！！！！！！注释里面的2 代表batch_size
        x1 = self.max_pool1_d1(x1)  # torch.Size([2, 4, 64])
        x1 = self.conv2_d1(x1)  # torch.Size([2, 8, 64])
        x1 = self.max_pool2_d1(x1)  # torch.Size([2, 8, 32])
        x1 = self.conv3_d1(x1)  # torch.Size([2,16,32])
        x1 = self.GAP(x1)  # torch.Size([2, 16, 1])
        #
        # x2 = self.conv1_d2(x2)  # torch.Size([2, 4, 128])
        # x2 = self.max_pool1_d2(x2)  # torch.Size([2, 4, 64])
        # x2 = self.conv2_d2(x2)  # torch.Size([2, 8, 64])
        # x2 = self.max_pool2_d2(x2)  # torch.Size([2, 8, 32])
        # x2 = self.conv3_d2(x2)  # torch.Size([2,16,32])
        # x2 = self.GAP(x2)  # torch.Size([2, 16, 1])
        #
        # x3 = self.conv1_d3(x3)  # torch.Size([2, 4, 128])
        # x3 = self.max_pool1_d3(x3)  # torch.Size([2, 4, 64])
        # x3 = self.conv2_d3(x3)  # torch.Size([2, 8, 64])
        # x3 = self.max_pool2_d3(x3)  # torch.Size([2, 8, 32])
        # x3 = self.conv3_d3(x3)  # torch.Size([2,16,32])
        # x3 = self.GAP(x3)  # torch.Size([2, 16, 1])

        # 全连接层前将通道数这一维度降取，原shape = [batch_size,channels,length] ----------> 现shape = [batch_size,channels*length]
        x1 = torch.flatten(x1, 1)  # x1 = x1.view(x1.size(0), -1) 等价于 x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)
        # x3 = torch.flatten(x3, 1)
        # print('-----------------------------------')
        # print(' torch.flatten(x1, 1)', x1.shape)
        # cat操作,将三个2*16-------------->一个2*48
        # x = torch.cat((x1), 1)
        # print('----------------------------------')
        # print('torch.cat((x1, x2,x3), 1)', x.shape)  # torch.Size([2, 48])
        # x = x1.view(x1.size(0), 1, -1)
        # # print('----------------------------------')
        # print('x.view(x.size(0), 1, -1)', x.shape)  # torch.Size([2,1, 48])
        #
        # x = self.self_attention_layers(x)
        # # print('self.self_attention_layers(x)',x.shape)
        x = x1.view(x1.size(0), -1)
        x = self.out(x)
        # print('----------------------------------')
        # print('x=', x)
        # print('x.shape',x.shape)

        return x


def train(model, loss_func, optimizer, train_loader, test_loader):
    model = model.to(device)
    loss_list_epoch = {'index': [], 'loss': [], 'test acr': [], 'train acr': []}
    for epoch in range(epochs):

        # adjust_learning_rate(optimizer, epoch, init_lr=lr)
        # if epoch % 10 == 0:
        #     for param_group in optimizer.param_groups:
        #         print(param_group['lr'])
        with tqdm(train_loader, total=len(train_loader)) as t:
            for batch_index, (train_x, train_y) in enumerate(t):
                # print(f'当前epoch:{epoch},batch_size:{batch_index + 1}')
                train_x = normalize(train_x)
                train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1],train_x.shape[2])
                train_x = train_x.to(device)
                label = train_y.to(device)
                model.train()
                output = model.forward(x1=train_x, x2=train_x, x3=train_x)
                loss = loss_func(output, label)
                loss_print = round(loss.detach().cpu().numpy() / 1, 4)
                # print('loss', loss)
                # 反向传播
                loss.backward()
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 梯度清零
                # 每个epoch打印一次
                if batch_index % print_batch == 0:
                    loss_list_epoch['index'].append(str(epoch) + '_' + str(batch_index / print_batch))
                    loss_list_epoch['loss'].append(loss_print)
                    test_acr = predict(model, test_loader, device)
                    train_acr = predict(model, train_loader, device)
                    loss_list_epoch['test acr'].append(test_acr)
                    loss_list_epoch['train acr'].append(train_acr)
                t.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                t.set_postfix(loss=loss_print, train_acr=train_acr, test_acr=test_acr)  # 🌟 3. 设置结尾
                time.sleep(0.00001)
    df = pd.DataFrame(loss_list_epoch)
    print(df)
    draw(df)


if __name__ == '__main__':
    # train_x1 = torch.randn(2, 1, 128)
    # train_x2 = torch.randn(2, 1, 128)
    # train_x3 = torch.randn(2, 1, 128)
    # print(train_x)

    model = MC_DCNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    train_data = get_trainData()
    test_data = get_testData()
    train(model, loss_func, optimizer, train_data, test_data)

