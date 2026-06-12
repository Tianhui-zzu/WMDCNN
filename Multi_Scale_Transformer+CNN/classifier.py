'hybrid attention-based transformer network with domain adversarial learning (HATN-DA)的分类器'
import torch
import torch.nn as nn
import torch.nn.functional as F


# 先简单的几层全连接层

class Classifier(nn.Module):
    def __init__(self, sensor_num=8, class_num=5, vector_dim=224,memory_f = None):
        super(Classifier, self).__init__()
        self.linear0 = nn.Linear(vector_dim, vector_dim // 2)
        self.linear1 = nn.Linear(vector_dim // 2, vector_dim // 4)
        self.linear2 = nn.Linear(vector_dim // 4, vector_dim // 8)
        # self.linear3 = nn.Linear(vector_dim // 8, vector_dim // 16)
        # 可以采取加如多维多卷积模型

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(sensor_num, class_num)
        self.memory = memory_f

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = self.GAP(x)
        # 暂存分类前的特征矩阵，便于呈现效果图
        if self.memory != None:
            temp = x.view(x.size(0), -1).detach().cpu().numpy()
            # print('x.type',type(x))
            # print('temp-------',temp)
            self.memory.append(temp)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


class ChannelAttention(nn.Module):
    # in_planes等于输入的通道数即传感器的个数
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
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
        print('avg_out + max_out',avg_out + max_out)
        return self.sigmoid(out)


# 卷积num*8*448特征向量
class Classifier_conv_new(nn.Module):
    def __init__(self, sensor_num=8, class_num=5, vector_dim=112,memory_f = None):
        super(Classifier_conv_new, self).__init__()

        self.channel_attention = ChannelAttention(in_planes=8)
        # 1*8*112
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.MaxPool2d(2))
        # 8*4*56
        self.conv2 = nn.Sequential(nn.Conv2d(8, 8, 3, 1, 1), nn.MaxPool2d(2))
        # 8*2*28
        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, 3, 1, 1), nn.MaxPool2d(2))
        # 8 * 1 * 14
        # self.GAP = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(8 * vector_dim // (8), 8 * vector_dim // (16))
        self.out = nn.Linear(8 * vector_dim // (16), class_num)
        self.memory = memory_f
    def forward(self, x):
        c_att_value = self.channel_attention(x)
        # print('x', x, '\n')
        x = c_att_value.mul(x)
        print('c_att_value', c_att_value, '\n')
        print('c_att_value.mul(x)', x, '\n')
        x = torch.unsqueeze(x, 1)
        # num, 1, 8, 128
        x = F.relu(self.conv1(x))
        # [num, 8, 4, 64]
        x = F.relu(self.conv2(x))
        # [num, 8, 2, 32]
        x = F.relu(self.conv3(x))
        # [num, 8, 1, 16]
        # x = x.view(x.size(0), -1, x.size(3))
        # --------------------------
        # [num, 8, 8]
        # [num, 8, 1]
        # x = x.view(x.size(0), x.size(1),-1)
        # x = self.GAP(x)
        x = x.view(x.size(0), -1)

        x = self.linear1(x)

        if self.memory != None:
            temp = x.view(x.size(0), -1).detach().cpu().numpy()
            # print('x.type',type(x))
            # print('temp-------',temp)
            self.memory.append(temp)
        x = self.out(x)
        # -----------------------
        return x
