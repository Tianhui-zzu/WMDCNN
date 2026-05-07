# 分类器
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn.functional import linear, normalize

mean = 0
std = 0.01


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
        return self.sigmoid(out)


# 融合了additive_angular_margin_loss的分类器
class PartialFC(torch.nn.Module):
    def __init__(self, feature_size=112 * 8, vector_dim=112, class_num=4, ppm_num=10, memory_f=None):
        super(PartialFC, self).__init__()
        self.feature_size = feature_size
        self.class_num = class_num
        # 生成一个权重矩阵,注意权重矩阵的形状为(class_num * ppm_num, feature_size),
        # 进行linear线性操作时候权重矩阵自动转置为(feature_size,class_num * ppm_num)
        self.channel_attention = ChannelAttention(in_planes=8)
        # 1*8*112
        self.linear0 = nn.Linear(vector_dim, vector_dim // 2)
        self.linear1 = nn.Linear(vector_dim // 2, vector_dim // 4)
        self.linear2 = nn.Linear(vector_dim // 4, vector_dim // 8)
        # 8*1*14
        self.weight = torch.nn.Parameter(torch.normal(mean, std, (class_num, feature_size // (1))))
        self.memory = memory_f

    def forward(self, feature_vector):
        # 通道注意力机制,三层卷积。将vectors(batch_size,8,112)->(batch_size,8,14)
        c_att_value = self.channel_attention(feature_vector)
        feature_vector = c_att_value.mul(feature_vector)
        # feature_vector = torch.unsqueeze(feature_vector, 1)
        # feature_vector = F.relu(self.linear0(feature_vector))
        # feature_vector = F.relu(self.linear1(feature_vector))
        # feature_vector = F.relu(self.linear2(feature_vector))
        # （batch_size,8，14）
        weight = self.weight
        batch_size = feature_vector.size(0)  # （batch_size,8*14）
        # 标准化,线性连接
        feature_vector = feature_vector.view(batch_size, -1)
        norm_feature = normalize(feature_vector)
        norm_weight_activated = normalize(weight)
        # 得出输出向量。
        out = linear(norm_feature, norm_weight_activated)
        if self.memory != None:
            temp = out.view(out.size(0), -1).detach().cpu().numpy()
            # print('x.type',type(x))
            # print('temp-------',temp)
            self.memory.append(temp)
        return out

    def change_label_form(self):
        return
