import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.fft as fft
from einops import rearrange, reduce, repeat
import numpy as np
from layers.Embed import *

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class Trend_Predit(nn.Module):
    def __init__(self, configs, scale_window=2, scale_num=3):
        super(Trend_Predit, self).__init__()
        self.configs = configs
        self.scale_window = scale_window
        self.scale_num = scale_num
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.seq_len // (scale_window ** i),
                    self.pred_len,
                )
                for i in range(scale_num+1)
            ]
        )
        self.projection_layer = nn.Linear(
            configs.d_model, configs.c_out, bias=True)
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        down_pool = torch.nn.AvgPool1d(self.scale_window).to(x_enc)
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.scale_num):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.scale_window, :])
            x_enc_ori = x_enc_sampling
            x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.scale_window, :]
        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list
        return x_enc, x_mark_enc
        # return x_enc


    def forward(self, x_enc, x_mark_enc):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_embed = [self.enc_embedding(x_enc[i], x_mark_enc[i]) for i in range(self.scale_num+1)]
        out_list = []
        for i, enc_out in zip(range(len(x_embed)), x_embed):
            dec_out = self.predict_layers[i](enc_out.permute(0,2,1)).permute(0,2,1)
            dec_out = self.projection_layer(dec_out)
            out_list.append(dec_out)
        out = torch.stack(out_list, dim=-1).sum(-1)
        return out


    # def forward(self, x_enc):
    #     x_enc = self.__multi_scale_process_inputs(x_enc)
    #     out_list = []
    #     for i, enc_out in zip(range(len(x_enc)), x_enc):
    #         dec_out = self.predict_layers[i](enc_out.permute(0,2,1)).permute(0,2,1)
    #         dec_out = self.projection_layer(dec_out)
    #         out_list.append(dec_out)
    #     out = torch.stack(out_list, dim=-1).sum(-1)
    #     return out

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        # self.decomp1 = series_decomp(moving_avg)
        # self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, t, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x, t,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return y, attn



class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, t, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for i, attn_layer in enumerate(self.attn_layers):
                x, attn = attn_layer(x, t[i], attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu


    def forward(self, x, cross, t, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x, t,
            attn_mask=x_mask
        )[0])
        y = x + self.dropout(self.cross_attention(
            x, cross, cross, t,
            attn_mask=cross_mask
        )[0])
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return y

class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, t, x_mask=None, cross_mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, cross, t[i], x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x



class IrregularPred(nn.Module):
    def __init__(self, configs):
        super(IrregularPred, self).__init__()
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        activation = configs.activation
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.predict = nn.Linear(configs.seq_len, configs.pred_len)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_enc):
        B, L, D = x_enc.shape
        x_out = x_enc.clone()
        for i in range(1, L):
            x_out[:, i, :] = x_enc[:, i, :] - x_enc[:, i-1, :]
        y = self.dropout(self.activation(self.conv1(x_out.transpose(-1, 1))))
        y = self.predict(y)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.projection(y)




