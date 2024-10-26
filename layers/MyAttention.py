import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os
from utils.masking import TriangularCausalMask, ProbMask
from layers.SelfAttention_Family import FullAttention
from layers.AutoCorrelation import AutoCorrelation

class MyAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(MyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, agents, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = keys.shape
        _, A, _, D = agents.shape
        scale = 1. / sqrt(E)

        scores1 = torch.einsum("blhe,bshe->bhls", agents, keys)
        A1 = self.dropout(torch.softmax(scale * scores1, dim=-1))
        V1 = torch.einsum("bhls,bshd->blhd", A1, values)

        scores2 = torch.einsum("blhe,bshe->bhls", queries, agents)
        A2 = self.dropout(torch.softmax(scale * scores2, dim=-1))
        V2 = torch.einsum("bhls,bshd->blhd", A2, V1)

        if self.output_attention:
            return (V2.contiguous(), A2)
        else:
            return (V2.contiguous(), None)


class MyAttentionLayer(nn.Module):
    def __init__(self, attn, d_model, n_heads, d_keys=None, d_values=None):
        super(MyAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.atten = attn
        self.full_attention = AutoCorrelation(mask_flag=False)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # 经过线性变换 shape保持不变
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.agent_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, agents, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        _, A, _ = agents.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        agents = self.agent_projection(agents).view(B, A, H, -1)

        out, attn = self.atten(
            queries,
            keys,
            values,
            agents,
            attn_mask
        )
        # out, attn = self.full_attention(
        #     queries,
        #     keys,
        #     values,
        #     attn_mask
        # )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn