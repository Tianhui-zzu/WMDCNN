'hybrid attention-based transformer network with domain adversarial learning (HATN-DA)的特征提取器'

import torch
import torch.nn as nn
import numpy as np
import math, copy

t_n =2 # transformer数量

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#   Layer&Norm
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


class Self_Attention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_q, dim_v):
        super(Self_Attention, self).__init__()
        self.k = nn.Linear(input_dim, dim_k, bias=False)
        self.q = nn.Linear(input_dim, dim_q, bias=False)
        self.v = nn.Linear(input_dim, dim_v, bias=False)
        self._norm_fact = 1 / (dim_k ** 0.5)
        self.mask = self.SubsequentMask(dim_v)  # input_dim=dim_k=dim_q=dim_v

    def SubsequentMask(self, size):
        """
        用于遮住序列的一些位置
        修改了解码器中的自注意力子层，以防止位置关注后续位置。
        这种掩蔽与输出嵌入偏移一个位置的事实相结合，确保了位置i的预测只能依赖小于位置i的已知输出
        :param size: (int)向量长度
        :return: (Tensor,bool)掩码后的矩阵，尺寸为[1,size,size]
        """
        attn_shape = (1, 1, size, size)
        # 返回函数的上三角矩阵，从k=1列开始
        subsequent_mask = np.triu(np.ones(attn_shape), k=0).astype('uint8')
        return torch.from_numpy(subsequent_mask).to(device)

    def forward(self, x):
        K = self.k(x)
        # print("K.shape",K.shape)
        Q = self.q(x)
        # print("Q.shape", Q.shape)
        V = self.v(x)
        # print("V.shape", V.shape)
        l = len(Q.shape)
        # print(Q.transpose(l - 1, l - 2).shape)
        atten = torch.matmul(Q.transpose(l - 1, l - 2), K) * self._norm_fact
        # if self.mask is not None:
        #     # print(self.mask == 0)
        #     atten.masked_fill_(self.mask == 0, -1e9)
        #     # print('atten before', atten)
        atten = torch.softmax(atten, dim=2)  # dim=2 即在第(B,C,H,W)上的第3个维度上归一化
        # print('atten', atten[0][0][0].sum())
        # print('atten.shape', atten.shape)
        # print('atten', atten)
        # 系数矩阵和原输入进行内积(向量积)
        output = torch.matmul(V, atten)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, head, input_dim, output_dim):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        # 多头注意力机制,将输入的词向量维度按头数分割
        self.d_k_in = input_dim // head
        self.d_k_out = output_dim // head
        # 假设input_dim和query、key、value的维度相同
        self.attention = Self_Attention(input_dim // head, output_dim // head, output_dim // head, output_dim // head)
        # 多头注意力机制需要在输出前再加入一次全连接
        self.linear = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.head, self.d_k_in).transpose(1, 2)
        # print('1---------x.shape', x.shape)
        x = self.attention(x)
        # print('2---------x.shape', x.shape)
        # 3) 拼接所有head然后线性变换
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.head * self.d_k_out)
        # print('3---------x.shape', x.shape)
        # 特征提取器的子部分-三层的前馈神经网络
        return self.linear(x)


class FFN_Clock(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(FFN_Clock, self).__init__()
        self.linear1 = nn.Linear(dim1, dim2)
        self.linear2 = nn.Linear(dim2, dim3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Weights_attetion(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(Weights_attetion, self).__init__()
        self.linear1 = nn.Linear(dim1, dim2)
        self.linear2 = nn.Linear(dim2, dim3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return torch.softmax(torch.tanh(x), dim=2)


# 将transformer中Encoder中的单注意力机制换成多头注意力机制,将其单独提取定义为一类，方便同一模块的叠加,多重复制
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.05, head=1):
        super(Encoder, self).__init__()
        self.attention1 = MultiHeadAttention(head, input_dim, output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = LayerNormalization(output_dim)
        self.ffn = FFN_Clock(output_dim, output_dim * 4, output_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm3 = LayerNormalization(output_dim)
        # 残差链接
        self.res = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.attention1(x) + self.res(x)
        x = self.dropout1(x)
        x = self.layer_norm1(x)
        x = self.ffn(x) + x
        x = self.dropout3(x)
        x = self.layer_norm3(x)
        return x


# 克隆函数,生成多个需要的module()
def clones(module, N):
    "生成N个相同的层."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 将5000采样点->1000采样点
# def down_sample(x):


class Feature_Extractor(nn.Module):
    def __init__(self, input_dim, out_dim, sensor_num=8, dropout=0.1, head=4, transformer_num=t_n):
        super(Feature_Extractor, self).__init__()
        # 通道数等于传感器的个数，输入的矩阵为(s)
        self.channels = sensor_num
        # self.embedding = Embeddings(5000, 512)
        # 5000->512，
        # self.down_sample = nn.MaxPool1d(kernel_size=2)
        self.Eembedding = nn.Linear(input_dim, out_dim, bias=True)
        '注意力第一层'
        self.Encoders = clones(Encoder(out_dim, out_dim, dropout, head), t_n)
        # self.Encoder1 = Encoder(out_dim, out_dim // 2, dropout, head)
        # self.Encoder2 = Encoder(out_dim // 2, out_dim // (2 * 2), dropout, head)
        # self.Encoder3 = Encoder(out_dim // (2 * 2), out_dim // (2 * 2 * 2), dropout, head)
        # 通道注意力机制（传感器个数）
        self.w_attention = Weights_attetion(self.channels, self.channels // 4, self.channels)

    # self.res1 = nn.Conv2d(1, 1, 1)
    # self.res2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        # print('x before down_sample shape', x.shape)
        # x = self.down_sample(x)
        # print('x after down_sample shape', x.shape)
        x = self.Eembedding(x)
        # 1
        for encoder in self.Encoders:
            x = encoder(x)
        # x = self.Encoder1(x)
        # x = self.Encoder2(x)
        # x = self.Encoder3(x)
        # 通道注意力机制
        # x_t = x.transpose(1, 2)
        # print('x_t.shape', x_t.shape)
        # print('self.w_attention(x_t)', self.w_attention(x_t)[0][0])
        # x = (self.w_attention(x_t) * x_t).transpose(1, 2)
        # print('x.shape', x.shape)
        return x
