import torch
import torch.nn as nn
import torch.nn.functional as F
from moudles import clones
from moudles import LayerNormalization, SubLayerConnection, MultiHeadedAttention, FeedForward


# 一整个编码器,由多个编码器子层组成
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNormalization(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, att, ff, dropout=0.05):
        '''
        :param size: 向量尺寸 一般设为512
        :param att: 多头注意力机制
        :param ff: 前馈连接网络
        :param dropout:
        '''
        super(EncoderLayer,self).__init__()
        self.size = size
        self.att = att
        self.ff = ff
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # 将注意力机制模块和输入向量传入子模块中,得到输出结果
        x = self.sublayer[0](x, lambda x: self.att(x, x, x, mask))
        return self.sublayer[1](x, self.ff)
