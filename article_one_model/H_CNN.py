import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim

G = 5
class ChannelAttention1d(nn.Module):
    # in_planes等于输入的通道数即传感器的个数
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class ChannelAttention2d(nn.Module):
    # in_planes等于输入的通道数即传感器的个数
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class H_CNN(nn.Module):
    def __init__(self):
        super(H_CNN, self).__init__()
        # (b,1,(40))
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,  #
                out_channels=20,  # 输出的通道数,等价于卷积核的个数.
                kernel_size=3,  # 卷积核的尺寸为1*3(1d)
                stride=1,  # 步长为1
                padding=1,  # 填充行数为1,由公式 padding=(kernel_size-1)/2(if padding=1),可保证经过卷积后图像特征图尺寸与原来保持一致,即仍为28*28
            ),
            nn.MaxPool1d(kernel_size=2),  #
            nn.Dropout(p=0.25)  # 25%概率将值置为0
        )
        # 第一层通道注意力机制
        self.channel_attention01 = ChannelAttention1d(20)
        # (b,1,(20,20))
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=3,  # 卷积核的尺寸为3*3(2d)
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25)
        )
        # 第二层通道注意力机制
        self.channel_attention02 = ChannelAttention2d(10)
        # (b,1,(10,10,10))
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=5,
                kernel_size=3,  # 卷积核的尺寸为3*3*3(3d)
                stride=1,
                padding=1,
            ),
            nn.MaxPool3d(kernel_size=2),  # 3d卷积
            nn.Dropout(p=0.25)
        )
        # (b,5,5,5,5)
        self.dense = nn.Linear(5 * 5 * 5 * 5, 10*G)
        self.out = nn.Linear(10 * G, G)

        # self.linear01 = nn.Linear(8,8)
        # self.out = nn.Linear(8,G)

    def forward(self, x):
        x = self.conv1(x)
        c_att_value01 = self.channel_attention01(x)
        x = c_att_value01.mul(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        # # # print('x.shape',x)
        x = self.conv2(x)
        # # c_att_value02 = self.channel_attention02(x)
        # # x = c_att_value02.mul(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # view方法类似于reshape方法,将tensor维度重新规划两维,(batch_size,32*7*7)
        x = self.dense(x)
        # # print('dense()后x.shape', x.shape)
        x = self.out(x)
        # print('全连接层后x.shape', x.shape)
        # print('全连接层后x', x.shape)

        # x = self.linear01(x)
        # x = self.out(x)
        return x

# 测试模型是否可以用

if __name__ == '__main__':
    t1 = torch.ones(4,1,40)
    # t1 = torch.LongTensor(20, 1, 8, 5000)
    model = H_CNN()
    final_out = model(t1)
    print('final_out :', final_out)
    print('final_out.shape:', final_out.shape)
