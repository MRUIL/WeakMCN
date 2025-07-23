import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.DCN.modules.deform_conv2d import DeformConv2d
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import numpy as np


# ------------------------------
# ----Visual Encoder Blocks-----
# ------------------------------

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class SPPLayer(nn.Module):
    def __init__(self):
        super(SPPLayer, self).__init__()

    def forward(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_4 = F.max_pool2d(x, 13, stride=1, padding=6)
        out = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        return out


class DropBlock(nn.Module):
    def __init__(self, block_size=7, keep_prob=0.9):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)

    def reset(self, block_size, keep_prob):
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)

    def calculate_gamma(self, x):
        return (1 - self.keep_prob) * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x):
        if (not self.training or self.keep_prob == 1):  # set keep_prob=1 to turn off dropblock
            return x
        if self.gamma is None:
            self.gamma = self.calculate_gamma(x)
        if x.type() == 'torch.cuda.HalfTensor':
            FP16 = True
            x = x.float()
        else:
            FP16 = False
        p = torch.ones_like(x) * (self.gamma)
        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)

        out = mask * x * (mask.numel() / mask.sum())

        if FP16:
            out = out.half()
        return out


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch // 2, 1, 1))
            resblock_one.append(add_conv(ch // 2, ch, 3, 1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class RFBblock(nn.Module):
    def __init__(self, in_ch, residual=False):
        super(RFBblock, self).__init__()
        inter_c = in_ch // 4
        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=2, padding=2)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=3, padding=3)
        )
        self.residual = residual

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        out = torch.cat((x_0, x_1, x_2, x_3), 1)
        if self.residual:
            out += x
        return out


class FeatureAdaption(nn.Module):
    def __init__(self, in_ch, out_ch, n_anchors, rfb=False, sep=False):
        super(FeatureAdaption, self).__init__()
        if sep:
            self.sep = True
        else:
            self.sep = False
            self.conv_offset = nn.Conv2d(in_channels=2 * n_anchors,
                                         out_channels=2 * 9 * n_anchors, groups=n_anchors, kernel_size=1, stride=1,
                                         padding=0)
            self.dconv = DeformConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1,
                                      padding=1, deformable_groups=n_anchors)
            self.rfb = None
            if rfb:
                self.rfb = RFBblock(out_ch)

    def forward(self, input, wh_pred):
        # The RFB block is added behind FeatureAdaption
        # For mobilenet, we currently don't support rfb and FeatureAdaption
        if self.sep:
            return input
        if self.rfb is not None:
            input = self.rfb(input)
        wh_pred_new = wh_pred.detach()
        offset = self.conv_offset(wh_pred_new)
        out = self.dconv(input, offset)
        return out


# --------------------------------
# ----Language Encoder Blocks-----
# --------------------------------

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.weight * (x - mean) / (std + self.eps) + self.bias


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.HIDDEN_SIZE // 2,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE,
            __C.HIDDEN_SIZE
        )

    def forward(self, x, x_mask=None):
        b, l, c = x.size()
        att = self.mlp(x).view(b, l, -1)
        x = x.view(b, l, self.__C.FLAT_GLIMPSES, -1)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x[:, :, i, :], dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        # print(scores.size(),mask.size())
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()
        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, y, y_mask, pos=None):
        q = k = self.with_pos_embed(y, pos)
        y = self.norm1(y + self.dropout1(
            self.mhatt(v=y, k=k, q=q, mask=y_mask)
        ))
        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# --------------------------------
# ----MultiScale Fusion Blocks----
# --------------------------------


def darknet_conv(in_ch, out_ch, ksize, stride=1, dilation_rate=1):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (dilation_rate * (ksize - 1) + 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False, dilation=dilation_rate))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage


class MultiScaleFusion(nn.Module):
    def __init__(self, v_planes=[256, 512, 1024], hiden_planes=512, scaled=True):
        super().__init__()
        self.up_modules = nn.ModuleList(
            [nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-2] + hiden_planes // 2, hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
            ),
                nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2) if scaled else nn.Sequential(),
                    darknet_conv(v_planes[-1], hiden_planes // 2, ksize=1),
                    darknet_conv(hiden_planes // 2, hiden_planes // 2, 3)
                )]
        )

        self.down_modules = nn.ModuleList(
            [nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(hiden_planes // 2 + v_planes[0], hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
            ),
                nn.Sequential(
                    nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                    darknet_conv(hiden_planes + v_planes[1], hiden_planes // 2, ksize=1),
                    darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
                )]
        )

        self.top_proj = darknet_conv(v_planes[-1] + hiden_planes // 2, hiden_planes, 1)
        self.mid_proj = darknet_conv(v_planes[1] + hiden_planes, hiden_planes, 1)
        self.bot_proj = darknet_conv(v_planes[0] + hiden_planes // 2, hiden_planes, 1)

    def forward(self, x):
        l, m, s = x
        # print(s)
        m = torch.cat([self.up_modules[1](s), m], 1)
        l = torch.cat([self.up_modules[0](m), l], 1)
        # out=self.out_proj(l)

        m = torch.cat([self.down_modules[0](l), m], 1)

        s = torch.cat([self.down_modules[1](m), s], 1)

        # top prpj and bot proj
        top_feat = self.top_proj(s)
        mid_feat = self.mid_proj(m)
        bot_feat = self.bot_proj(l)
        return [bot_feat, mid_feat, top_feat]


class SimpleFusion(nn.Module):
    def __init__(self, v_planes=1024, q_planes=1024, out_planes=1024):
        super().__init__()
        self.v_proj = nn.Sequential(
            nn.Conv2d(v_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
        self.q_proj = nn.Sequential(
            nn.Conv2d(q_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, y):
        x = self.v_proj(x)
        y = self.q_proj(y.unsqueeze(2).unsqueeze(2))
        return self.norm(x * y)


class CollectDiffuseAttention(nn.Module):
    ''' CollectDiffuseAttention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_c = nn.Dropout(attn_dropout)
        self.dropout_d = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, kc, kd, v, mask=None):
        '''
        q: n*b,1,d_o
        kc: n*b,h*w,d_o
        kd: n*b,h*w,d_o
        v: n*b,h*w,d_o
        '''

        attn_col = torch.bmm(q, kc.transpose(1, 2))  # n*b,1,h*w
        attn_col_logit = attn_col / self.temperature
        attn_col = self.softmax(attn_col_logit)
        attn_col = self.dropout_c(attn_col)
        attn = torch.bmm(attn_col, v)  # n*b,1,d_o

        attn_dif = torch.bmm(kd, q.transpose(1, 2))  # n*b,h*w,1
        attn_dif_logit = attn_dif / self.temperature
        attn_dif = torch.sigmoid(attn_dif_logit)
        attn_dif = self.dropout_d(attn_dif)
        output = torch.bmm(attn_dif, attn)
        return output, attn_col_logit.squeeze(1)


class GaranAttention(nn.Module):
    ''' GaranAttention module '''

    def __init__(self, d_q, d_v, n_head=2, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q
        self.d_v = d_v
        self.d_k = d_v
        self.d_o = d_v
        d_o = d_v

        self.w_qs = nn.Linear(d_q, d_o)
        self.w_kc = nn.Conv2d(d_v, d_o, 1)
        self.w_kd = nn.Conv2d(d_v, d_o, 1)
        self.w_vs = nn.Conv2d(d_v, d_o, 1)
        self.w_m = nn.Conv2d(d_o, 1, 3, 1, padding=1)
        self.w_o = nn.Conv2d(d_o, d_o, 1)

        self.attention = CollectDiffuseAttention(temperature=np.power(d_o // n_head, 0.5))
        self.layer_norm = nn.BatchNorm2d(d_o)
        self.layer_acti = nn.LeakyReLU(0.1, inplace=True)
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, v, mask=None):
        d_k, d_v, n_head, d_o = self.d_k, self.d_v, self.n_head, self.d_o

        sz_b, c_q = q.size()
        sz_b, c_v, h_v, w_v = v.size()
        # print(v.size())
        residual = v

        q = self.w_qs(q)
        kc = self.w_kc(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)
        kd = self.w_kd(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)
        v = self.w_vs(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)
        q = q.view(sz_b, n_head, 1, d_o // n_head)
        # v=v.view(sz_b,h_v*w_v,n_head,c_v//n_head)

        q = q.view(-1, 1, d_o // n_head)  # (n*b) x lq x dk
        kc = kc.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        kd = kd.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        v = v.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lv x dv

        output, m_attn = self.attention(q, kc, kd, v)
        # n * b, h * w, d_o
        output = output.view(sz_b, n_head, h_v, w_v, d_o // n_head)
        output = output.permute(0, 1, 4, 3, 2).contiguous().view(sz_b, -1, h_v, w_v)  # b x lq x (n*dv)
        m_attn = m_attn.view(sz_b, n_head, h_v * w_v)
        # m_attn=m_attn.mean(1)

        # residual connect
        output = self.w_o(output)
        attn = output
        m_attn = self.w_m(attn).view(sz_b, h_v * w_v)
        output = self.layer_norm(output)
        output = output + residual
        output = self.layer_acti(output)

        # output = self.dropout(self.fc(output))

        return output, m_attn, attn
