import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.Myformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer, my_Layernorm, Trend_Predit, IrregularPred
from layers.MyAttention import MyAttentionLayer, MyAttention
from utils.RevIN import RevIN
from layers.ScaleDecompAttention import *
from layers.SeriesDecomp import SeriesDecomp

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len# 输入长度
        self.label_len = configs.label_len# label长度
        self.pred_len = configs.pred_len# 预测长度
        self.output_attention = configs.output_attention

        self.encoder_num = configs.e_layers
        self.decoder_num = configs.e_layers
        # 多尺度分解注意力
        self.scale_size = configs.down_sampling_window
        self.scale_windows = [configs.down_sampling_window ** (i+1) for i in range(self.encoder_num)]
        self.scale_num = [self.seq_len // size for size in self.scale_windows]
        self.scale_branch = len(self.scale_num)

        self.trend_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.irregular_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.revin = RevIN(0, affine=False)

        self.decomp = SeriesDecomp(configs)

        self.multi_scale = nn.ModuleList(
            [MultiScaleDecompLayer(ScaleDecompAttention(attention_dropout=configs.dropout),
                                  scale_size=self.scale_windows[i],
                                   scale_num=self.scale_num[i],
                                   d_model=configs.d_model,
                                   n_heads=configs.n_heads) for i in range(self.scale_branch)]
        )

        self.trend_predit = Trend_Predit(configs, scale_window=self.scale_size, scale_num=configs.down_sampling_layers)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MyAttentionLayer(
                        MyAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        d_model=configs.d_model, n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    MyAttentionLayer(
                        MyAttention(
                            True, configs.factor, attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    MyAttentionLayer(
                        MyAttention(
                            False, configs.factor, attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    c_out=configs.c_out,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.irregular_predit = IrregularPred(configs)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = self.revin(x_enc, 'norm')

        seasonal, trend, irregular = self.decomp(x_enc.unsqueeze(-1))

        # 季节预测输入初始化
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init = torch.cat([seasonal[:, -self.label_len:, :], zeros], dim=1)

        # 趋势预测
        trend_part = self.trend_predit(trend, x_mark_enc)

        # 季节预测
        enc_out = self.enc_embedding(seasonal, x_mark_enc)
        embed_out = [self.multi_scale[i](enc_out) for i in range(self.scale_branch)]
        enc_out, attns = self.encoder(enc_out, embed_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part = self.decoder(dec_out, enc_out, embed_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # irregular预测
        irregular_init = self.irregular_embedding(irregular, x_mark_enc)
        irregular_part = self.irregular_predit(irregular_init)

        # final
        dec_out = trend_part + seasonal_part[:, -self.pred_len:, :] + irregular_part
        dec_out = self.revin(dec_out, 'denorm')
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, L, D]


