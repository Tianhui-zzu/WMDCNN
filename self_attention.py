'自注意力机制'
import torch
import torch.nn as nn


class Self_Attention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_q, dim_v):
        super(Self_Attention, self).__init__()
        self.k = nn.Linear(input_dim, dim_k, bias=False)
        self.q = nn.Linear(input_dim, dim_q, bias=False)
        self.v = nn.Linear(input_dim, dim_v, bias=False)
        self._norm_fact = 1 / (dim_k ** 0.5)

    def forward(self, x):
        K = self.k(x)
        Q = self.q(x)
        V = self.v(x)
        print(Q.shape)
        print(Q.transpose(3, 2).shape)
        # 求出自注意力系数 transpose(-2,-1)
        atten = torch.softmax(torch.matmul(K, Q.transpose(3, 2)) * self._norm_fact,
                              dim=3)  # dim=3 ,torch.Size([2, 1, 8, 8])即在第4个维度上归一化
        print('atten.shape', atten.shape)
        print('atten', atten)
        # 系数矩阵和原输入进行内积(向量积)
        output = torch.matmul(atten, V)
        return output


torch.random.manual_seed(420)
#  batch,通道数,尺寸
X = torch.randn(2, 1, 8, 5000)
Attetion = Self_Attention(5000, 16, 16, 32)
Y = Attetion(X)

print(Y)
