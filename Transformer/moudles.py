from turtle import clone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter

import math, copy, time


def Attention(query, key, value, mask=None, dropout=None):
    "计算'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # attention得分计算，key要转置一下
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def SubsequentMask(size):
    """
    用于遮住序列的一些位置
    修改了解码器中的自注意力子层，以防止位置关注后续位置。
    这种掩蔽与输出嵌入偏移一个位置的事实相结合，确保了位置i的预测只能依赖小于位置i的已知输出
    :param size: (int)向量长度
    :return: (Tensor,bool)掩码后的矩阵，尺寸为[1,size,size]
    """
    attn_shape = (1, size, size)
    # 返回函数的上三角矩阵，从k=1列开始
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



# 克隆函数,生成多个需要的module()
def clones(module, N):
    "生成N个相同的层."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


# 残差连接
class SubLayerConnection(nn.Module):
    '''
    先归一化在连接,和连接后在归一化在多个模型循环连接的情况,效果一致
    '''

    def __init__(self, size, dropout=0.05):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNormalization(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        '''

        :param x: 原始输入向量
        :param sublayer: 可能是前馈连接层、多头注意力层
        :return:
        '''
        return self.norm(x + self.dropout(sublayer(x)))


class LayerNormalization(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        '''Applies layer normalization.

              Args:
                epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        '''
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

if __name__ == "__main__":
    # 根据位置添加正弦波
    plt.figure(figsize=(15, 5))
    # 设模型字嵌入维度为20
    pe = PositionalEncoding(20, 0)
    # 执行PE的前向传播，输入张量尺寸为[1, 100, 20]
    y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    # 随便画出几个维度的位置编码
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()

# 前馈连接层
class FeedForward(nn.Module):
    "实现FFN"

    def \
            __init__(self, d1, d2, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d1, d2)
        self.w_2 = nn.Linear(d2, d1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# 多头注意力层
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 假设 d_v 总是等于 d_k
        self.d_k = d_model // h
        self.h = h
        # 一共克隆了4个线性连接层,前3个是用来生成对应query、key、value的矩阵，最后一个用于线性映射
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "复现上图"
        if mask is not None:
            # 给所有h个heads应用相同的mask.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 对每个batch进行线性投影得到相应向量
        # 这里用的是全连接层实现，全连接层的权重其实就对应不同的Q,K,V矩阵啦~
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) 每个batch使用注意力
        x, self.attn = Attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 拼接所有head然后线性变换
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    # Press the green button in the gutter to run the script.
#
# if __name__ == '__main__':
#     # t = torch.Tensor(500, 200)
#     # # print(Parameter(t))
#     # t = torch.Tensor([[i] for i in range(9)])
#     # print(t)
#     t = torch.LongTensor(500, 200)
#     print('t', t)
#     e = Embeddings(200, 30)
#     f = e.forward(t)
#     print('f', f)
#     print('f.size()', f.size())
#     pe = PositionalEncoding(f)
#     print(pe.forward(f))
