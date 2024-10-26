import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.fft as fft
from einops import rearrange, reduce, repeat
import numpy as np
from layers.Embed import *

# 趋势分解
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=[4, 8, 12]):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean



# 季节分解
class FourierLayer(nn.Module):

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""

        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)# 对时间维度进行fft

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]# 这个不包含最后一个频率
            f = fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))# (mesh_a, mesh_b)表示坐标
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        T = x.size(1)

        dft_mat = fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = np.exp(2 * math.pi * 1j / T)
        idft_mat = (np.power(omega, i * j) / T).cfloat()

        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(torch.arange(x.size(0)), torch.arange(x.size(2)))

        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b o t d -> b d o t')


class SeriesDecomp(nn.Module):
    def __init__(self, configs):
        super(SeriesDecomp, self).__init__()
        self.configs = configs
        # self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])
        self.trend_model = Trend_Model(alpha=0.8, beta=0.2)
        self.season_model = FourierLayer(pred_len=0, k=3)

    def forward(self, x_enc):
        x_enc = x_enc[:, :, :, 0]
        _, trend = self.trend_model(x_enc)
        seasonality, _ = self.season_model(x_enc)
        irregular = x_enc - seasonality - trend
        return seasonality, trend, irregular



class Trend_Model(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super(Trend_Model, self).__init__()
        self.alpha = alpha
        # self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.beta = beta
        # self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)
    def forward(self, x):
        # 将时间维度放在最后
        x = x.transpose(1, 2)
        # 初始化 smoothed_series 和 trend
        smoothed_series = x.new_zeros(x.shape)
        trend = x.new_zeros(x[:, :, 1:].shape)
        # 计算 smoothed_series 和 trend
        smoothed_series[:, :, 0] = x[:, :, 0]
        trend[:, :, 0] = x[:, :, 1] - x[:, :, 0]
        for t in range(1, x.shape[2]):
            smoothed_series[:, :, t] = self.alpha * x[:, :, t] + (1 - self.alpha) * (smoothed_series[:, :, t - 1] + trend[:, :, t - 1])
            if t < x.shape[2] - 1:
                trend[:, :, t] = self.beta * (smoothed_series[:, :, t] - smoothed_series[:, :, t - 1]) + (1 - self.beta) * trend[:, :, t - 1]
        # 将时间维度放回中间位置并返回
        return None, smoothed_series.transpose(1, 2)