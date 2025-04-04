import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.TwoStageAttention import FullAttention, AttentionLayer, TwoStageAttentionLayer
from math import ceil



class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    '''

    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, \
                 seg_num=10, factor=10):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None  # Encoder 第一层不进行合并

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, \
                                                             d_ff, dropout))

    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    '''
    The Encoder of Crossformer.
    '''

    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout,
                 in_seg_num=10, factor=10):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout, \
                                              in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout, \
                                                  ceil(in_seg_num / win_size ** i), factor))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x





class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''

    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num=10, factor=10):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, \
                                                     d_ff, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''

        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')

        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp = self.cross_attention(
            x, cross, cross,
        )
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_output, layer_predict


class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''

    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout, \
                 router=False, out_seg_num=10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, \
                                                   out_seg_num, factor))

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)

        return final_predict

