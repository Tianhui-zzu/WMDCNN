'自注意力层'
import torch.nn as nn
import torch


# 子模块 注意力
class Attention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_q, dim_v):
        super(Attention, self).__init__()
        self.k = nn.Linear(input_dim, dim_k, bias=False)
        self.q = nn.Linear(input_dim, dim_q, bias=False)
        self.v = nn.Linear(input_dim, dim_v, bias=False)
        self._norm_fact = 1 / (dim_k ** 0.5)

    def forward(self, x):
        K = self.k(x)
        Q = self.q(x)
        V = self.v(x)
        # print(Q.shape)
        # print(Q.transpose(2, 1))
        # 求出自注意力系数 transpose(-2,-1)
        atten = torch.softmax(torch.matmul(K, Q.transpose(2, 1)) * self._norm_fact, dim=1)  # dim=1,K*Q在维度为1(即列上进行归一化)
        # 系数矩阵和原输入进行内积(向量积)
        output = torch.matmul(atten, V)
        return output


# 子模块 多层感知机
class MLP(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, dim5):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(dim1, dim2, bias=True)
        self.gelu1 = nn.GELU()
        self.linear2 = nn.Linear(dim2, dim3, bias=True)
        self.gelu2 = nn.GELU()
        self.linear3 = nn.Linear(dim3, dim4, bias=True)
        self.gelu3 = nn.GELU()
        self.linear4 = nn.Linear(dim4, dim5, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu1(x)
        x = self.linear2(x)
        x = self.gelu2(x)
        x = self.linear3(x)
        x = self.gelu3(x)
        x = self.linear4(x)
        return x


class Self_Attention_Layers(nn.Module):
    def __init__(self):
        super(Self_Attention_Layers, self).__init__()
        self.layer_norm1 = nn.LayerNorm([1, 16])
        self.attention = Attention(16, 16, 16, 16)
        self.layer_norm2 = nn.LayerNorm([1, 16])
        self.mlp = MLP(16, 32, 64, 128, 16)
        # 残差核
        self.conv1 = nn.Conv1d(1, 1, 1)
        self.conv2 = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        # 残差核x1
        x1 = self.conv1(x)
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = x + x1
        # 残差核x2
        x2 = self.conv2(x)
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + x2
        return x

# x = torch.randn(2, 1, 16)
#
# Model = Self_Attention_Layers()
# x = Model(x)
# print('Self_Attention_Layers(x)', x)
