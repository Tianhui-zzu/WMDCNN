import torch
import torch.nn as nn
import torch.nn.functional as F
from moudles import clones
from moudles import LayerNormalization, SubLayerConnection, MultiHeadedAttention, FeedForward


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNormalization(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)



# 与编码器类似，在每个子层周围使用残差连接，然后进行层归一化。
class DecoderLayer(nn.Module):
    "DecoderLayer由self-attn, src-attn和 feed forward 组成(后面定义)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # masked Multi-Head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Multi-Head Attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # feed forward
        return self.sublayer[2](x, self.feed_forward)
