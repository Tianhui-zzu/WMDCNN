import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_decoder import EncoderDecode, Generator
from decoder import Decoder, DecoderLayer
from encoder import Encoder, EncoderLayer
from moudles import MultiHeadedAttention, FeedForward, Embeddings
from moudles import PositionalEncoding


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=512 * 4, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout=dropout)
    model = EncoderDecode(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                          Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                          nn.Sequential(Embeddings(src_vocab, d_model), c(position)),
                          nn.Sequential(Embeddings(tgt_vocab, d_model), c(position)),
                          Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
