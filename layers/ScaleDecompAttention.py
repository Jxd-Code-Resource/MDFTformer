import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os
from utils.masking import TriangularCausalMask, ProbMask

class ScaleDecompAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(ScaleDecompAttention, self).__init__()
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V


class MultiScaleDecompLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, scale_size, scale_num, d_keys=None, d_values=None):
        super(MultiScaleDecompLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.scale_size = scale_size
        self.scale_num = scale_num

        # 将序列分成num段，就需要num个可学习的矩阵
        self.intra_embeddings = nn.Parameter(torch.rand(scale_num, 1, 1, d_model),
                                             requires_grad=True)
        # 将序列分成num段，对应生成Q矩阵就需要num个线性变换
        self.embeddings_generator = nn.ModuleList([nn.Sequential(*[
            nn.Linear(d_model, d_model)]) for _ in range(scale_num)])

    def forward(self, x):
        new_x = x
        batch_size = x.size(0)
        scale_out_concat = None
        for i in range(self.scale_num):
            t = x[:, i * self.scale_size:(i + 1) * self.scale_size, :]
            B, L, _ = t.shape
            H = self.n_heads
            # 生成可学习矩阵Q
            intra_emb = self.intra_embeddings[i].expand(batch_size, -1, -1)

            queries = self.query_projection(intra_emb).view(B, 1, H, -1)
            keys = self.key_projection(t).view(B, L, H, -1)
            values = self.value_projection(t).view(B, L, H, -1)

            out = self.inner_attention(queries, keys, values)
            if scale_out_concat is None:
                scale_out_concat =out
            else:
                scale_out_concat = torch.cat((scale_out_concat, out), dim=1)
        return scale_out_concat.view(batch_size, self.scale_num, -1)
