import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
import math
import numpy as np
from itertools import combinations
from einops import rearrange
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import random
NUM_SAMPLES = 1
np.random.seed(83)
torch.manual_seed(83)
torch.cuda.manual_seed(83)
torch.cuda.manual_seed_all(83)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """ Implement the PE function. """

    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        # pe is of shape max_len(5000) x 2048(last layer of FC)
        pe = torch.zeros(max_len, d_model)
        # position is of shape 5000 x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class mSEModule(nn.Module):
    def __init__(self, channel, n_segment=8):
        super(mSEModule, self).__init__()
        self.channel = channel
        self.reduction = 8
        self.n_segment = n_segment
        self.temperature = self.channel // self.reduction
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pe = PositionalEncoding(self.channel // self.reduction, 0.1, max_len=49)
        self.conv = nn.Conv2d(in_channels=self.channel,
                              out_channels=self.channel,
                              kernel_size=3, padding=1, groups=self.channel, bias=False)
        self.conv1 = nn.Conv2d(in_channels=self.channel,
                               out_channels=self.channel // self.reduction,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        self.conv2 = nn.Conv2d(in_channels=self.channel // self.reduction,
                               out_channels=self.channel // self.reduction,
                               kernel_size=3, padding=1, groups=self.channel // self.reduction, bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_backward2 = nn.AvgPool2d(kernel_size=2, stride=2)  # nn.AdaptiveMaxPool2d(1)

        self.pad1_forward = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)

        self.conv3 = nn.Conv2d(in_channels=self.channel // self.reduction,
                               out_channels=self.channel // self.reduction, kernel_size=1, bias=False)

        self.conv3_smallscale2 = nn.Conv2d(in_channels=self.channel // self.reduction,
                                           out_channels=self.channel // self.reduction, padding=1, kernel_size=3,
                                           bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        self.conv3_smallscale4 = nn.Conv2d(in_channels=self.channel // self.reduction,
                                           out_channels=self.channel // self.reduction, padding=1, kernel_size=3,
                                           bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        bottleneck = self.conv1(x)  # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w

        t_fea_forward, _ = reshape_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        _, t_fea_backward = reshape_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view(
            (-1, self.n_segment) + conv_bottleneck.size()[1:])  # n, t, c//r, h, w
        _, tPlusone_fea_forward = reshape_conv_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w
        tPlusone_fea_backward, _ = reshape_conv_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        diff_fea_forward = tPlusone_fea_forward - t_fea_forward  # n, t-1, c//r, h, w
        diff_fea_backward = tPlusone_fea_backward - t_fea_backward  # n, t-1, c//r, h, w
        diff_fea_pluszero_forward = F.pad(diff_fea_forward, self.pad1_forward, mode="constant",
                                          value=0)  # n, t, c//r, h, w
        diff_fea_pluszero_forward = diff_fea_pluszero_forward.view(
            (-1,) + diff_fea_pluszero_forward.size()[2:])  # nt, c//r, h, w
        diff_fea_pluszero_backward = F.pad(diff_fea_backward, self.pad1_backward, mode="constant",
                                           value=0)  # n, t, c//r, h, w
        diff_fea_pluszero_backward = diff_fea_pluszero_backward.view(
            (-1,) + diff_fea_pluszero_backward.size()[2:])  # nt, c//r, h, w
        y_forward_smallscale2 = self.avg_pool_forward2(diff_fea_pluszero_forward)  # nt, c//r, h//2, w//2
        y_backward_smallscale2 = self.avg_pool_backward2(diff_fea_pluszero_backward)  # nt, c//r, h//2, w//2

        y_forward_smallscale4 = diff_fea_pluszero_forward
        y_backward_smallscale4 = diff_fea_pluszero_backward
        y_forward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_forward_smallscale2))
        y_backward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_backward_smallscale2))

        y_forward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_forward_smallscale4))
        y_backward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_backward_smallscale4))

        y_forward_smallscale2 = F.interpolate(y_forward_smallscale2, diff_fea_pluszero_forward.size()[2:])
        y_backward_smallscale2 = F.interpolate(y_backward_smallscale2, diff_fea_pluszero_backward.size()[2:])
        # y_f:(40, 128, 7, 7), y_b:(40, 128, 7, 7)
        y_forward = self.conv3(
            1.0 / 3.0 * diff_fea_pluszero_forward + 1.0 / 3.0 * y_forward_smallscale2 + 1.0 / 3.0 * y_forward_smallscale4)  # nt, c, 1, 1
        y_backward = self.conv3(
            1.0 / 3.0 * diff_fea_pluszero_backward + 1.0 / 3.0 * y_backward_smallscale2 + 1.0 / 3.0 * y_backward_smallscale4)  # nt, c, 1, 1
        # (40,128,49)->(40,49,128)
        y_forward = self.pe(y_forward.reshape(y_forward.size()[:-2] + (-1,)).permute(0, 2, 1))
        # (40,128,49)->(40,49,128)
        y_backward = self.pe(y_backward.reshape(y_backward.size()[:-2] + (-1,)).permute(0, 2, 1))

        attn1 = torch.bmm(y_forward, y_forward.transpose(1, 2))
        attn2 = torch.bmm(y_backward, y_backward.transpose(1, 2))
        attn1 = attn1 / np.power(self.temperature, 0.5)
        attn2 = attn2 / np.power(self.temperature, 0.5)
        attn1 = self.softmax(attn1)
        attn2 = self.softmax(attn2)
        attn1 = self.dropout(attn1)
        attn2 = self.dropout(attn2)
        # (40, 2048, 7, 7)
        v = self.conv(x)
        # (40, 2048, 49)->(20,49,2048)
        v = v.reshape(v.size()[:-2] + (-1,)).permute(0, 2, 1)
        activate_v1 = torch.bmm(attn1, v)
        activate_v2 = torch.bmm(attn2, v)
        # (40, 49, 2048)
        activate = 0.5 * activate_v1 + 0.5 * activate_v2
        activate = activate.permute(0, 2, 1).reshape(-1, self.channel, 7, 7)
        output = x + activate * self.gamma
        return output


class MLP_Mix_1(nn.Module):
    def __init__(self, channel, n_segment=8):
        super(MLP_Mix_1, self).__init__()
        self.channel = channel
        self.n_segment = n_segment
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc_t1 = nn.Linear(n_segment, n_segment)
        self.fc_t2 = nn.Linear(n_segment, n_segment)

        self.fc_d1 = nn.Linear(channel, channel)
        self.fc_d2 = nn.Linear(channel, channel)

    def forward(self, x):
        # x.shape is (batch*8, 2048, 7, 7)
        H = x.size()[-1]
        W = H
        x = x.reshape(-1, self.n_segment, self.channel, H*W)
        x = x.permute(0, 3, 2, 1).contiguous()  # (batch, 49, 2048, 8)
        x = x + self.fc_t2(self.relu(self.fc_t1(x)))
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, 8, 49, 2048)
        x = x + self.fc_d2(self.relu(self.fc_d1(x)))
        x = x.permute(0, 1, 3, 2).reshape(-1, self.n_segment, self.channel, H, W)
        # (batch, 8, 2048, 7, 7)
        return x


class spa_mix_2(nn.Module):
    def __init__(self, channel, n_segment=8, num_patches=49):
        super(spa_mix_2, self).__init__()
        self.channel = channel
        self.num_patches = num_patches
        self.reduction = 8
        self.n_segment = n_segment
        self.temperature = self.channel // self.reduction
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.norm_qk = nn.LayerNorm(self.channel // self.reduction)
        self.conv = nn.Conv2d(in_channels=self.channel,
                              out_channels=self.channel,
                              kernel_size=3, padding=1, groups=self.channel, bias=False)
        self.conv1 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel // self.reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        self.conv2 = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel // self.reduction, kernel_size=3, padding=1, groups=self.channel // self.reduction, bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU(inplace=True)
        self.avg_pool_backward2 = nn.AvgPool2d(kernel_size=2, stride=2)  # nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pad1_forward = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)

        self.conv3 = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel // self.reduction, kernel_size=1, bias=False)

        self.conv3_smallscale2 = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel // self.reduction, padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        self.conv3_smallscale4 = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel // self.reduction, padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, support, query):
        bottleneck_s = self.conv1(support)      # nt, c//r, h, w
        bottleneck_s = self.bn1(bottleneck_s)   # nt, c//r, h, w
        bottleneck_q = self.conv1(query)        # nt, c//r, h, w
        bottleneck_q = self.bn1(bottleneck_q)   # nt, c//r, h, w

        reshape_bottleneck_s = bottleneck_s.view((-1, self.n_segment) + bottleneck_s.size()[1:])  # 25, 8, 256, 7, 7
        reshape_bottleneck_q = bottleneck_q.view((-1, self.n_segment) + bottleneck_q.size()[1:])  # 20, 8, 256, 7, 7
        query_num = reshape_bottleneck_q.shape[0]      # 20
        support_num = reshape_bottleneck_s.shape[0]    # 25
        t_fea_forward_s, _ = reshape_bottleneck_s.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        _, t_fea_backward_s = reshape_bottleneck_s.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w
        t_fea_forward_q, _ = reshape_bottleneck_q.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        _, t_fea_backward_q = reshape_bottleneck_q.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w

        conv_bottleneck_s = self.conv2(bottleneck_s)  # nt, c//r, h, w
        conv_bottleneck_q = self.conv2(bottleneck_q)  # nt, c//r, h, w
        reshape_conv_bottleneck_s = conv_bottleneck_s.view((-1, self.n_segment) + conv_bottleneck_s.size()[1:])  # n, t, c//r, h, w
        reshape_conv_bottleneck_q = conv_bottleneck_q.view((-1, self.n_segment) + conv_bottleneck_q.size()[1:])  # n, t, c//r, h, w

        _, tPlusone_fea_forward_s = reshape_conv_bottleneck_s.split([1, self.n_segment - 1], dim=1)   # n, t-1, c//r, h, w
        tPlusone_fea_backward_s, _ = reshape_conv_bottleneck_s.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        diff_fea_forward_s = tPlusone_fea_forward_s - t_fea_forward_s           # n, t-1, c//r, h, w
        diff_fea_backward_s = tPlusone_fea_backward_s - t_fea_backward_s        # n, t-1, c//r, h, w
        _, tPlusone_fea_forward_q = reshape_conv_bottleneck_q.split([1, self.n_segment - 1], dim=1)   # n, t-1, c//r, h, w
        tPlusone_fea_backward_q, _ = reshape_conv_bottleneck_q.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        diff_fea_forward_q = tPlusone_fea_forward_q - t_fea_forward_q           # n, t-1, c//r, h, w
        diff_fea_backward_q = tPlusone_fea_backward_q - t_fea_backward_q        # n, t-1, c//r, h, w

        diff_fea_forward_s = F.pad(diff_fea_forward_s, self.pad1_forward, mode="constant", value=0)     # n, t, c//r, h, w
        diff_fea_forward_s = diff_fea_forward_s.view((-1,) + diff_fea_forward_s.size()[2:])             # nt, c//r, h, w
        diff_fea_backward_s = F.pad(diff_fea_backward_s, self.pad1_backward, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_backward_s = diff_fea_backward_s.view((-1,) + diff_fea_backward_s.size()[2:])          # nt, c//r, h, w
        diff_fea_forward_q = F.pad(diff_fea_forward_q, self.pad1_forward, mode="constant", value=0)     # n, t, c//r, h, w
        diff_fea_forward_q = diff_fea_forward_q.view((-1,) + diff_fea_forward_q.size()[2:])             # nt, c//r, h, w
        diff_fea_backward_q = F.pad(diff_fea_backward_q, self.pad1_backward, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_backward_q = diff_fea_backward_q.view((-1,) + diff_fea_backward_q.size()[2:])          # nt, c//r, h, w

        # support
        y_forward_smallscale2_s = self.avg_pool_forward2(diff_fea_forward_s)        # nt, c//r, h//2, w//2
        y_backward_smallscale2_s = self.avg_pool_backward2(diff_fea_backward_s)     # nt, c//r, h//2, w//2
        y_forward_smallscale4_s = diff_fea_forward_s
        y_backward_smallscale4_s = diff_fea_backward_s
        y_forward_smallscale2_s = self.bn3_smallscale2(self.conv3_smallscale2(y_forward_smallscale2_s))
        y_backward_smallscale2_s = self.bn3_smallscale2(self.conv3_smallscale2(y_backward_smallscale2_s))
        y_forward_smallscale4_s = self.bn3_smallscale4(self.conv3_smallscale4(y_forward_smallscale4_s))
        y_backward_smallscale4_s = self.bn3_smallscale4(self.conv3_smallscale4(y_backward_smallscale4_s))
        y_forward_smallscale2_s = F.interpolate(y_forward_smallscale2_s, diff_fea_forward_s.size()[2:])
        y_backward_smallscale2_s = F.interpolate(y_backward_smallscale2_s, diff_fea_backward_s.size()[2:])
        # (25*8, 256, 7, 7)
        y_forward_s = self.conv3(1.0 / 3.0 * diff_fea_forward_s + 1.0 / 3.0 * y_forward_smallscale2_s + 1.0 / 3.0 * y_forward_smallscale4_s)      # nt, c//r, 7, 7
        y_backward_s = self.conv3(1.0 / 3.0 * diff_fea_backward_s + 1.0 / 3.0 * y_backward_smallscale2_s + 1.0 / 3.0 * y_backward_smallscale4_s)  # nt, c//r, 7, 7

        # query
        y_forward_smallscale2_q = self.avg_pool_forward2(diff_fea_forward_q)        # nt, c//r, h//2, w//2
        y_backward_smallscale2_q = self.avg_pool_backward2(diff_fea_backward_q)     # nt, c//r, h//2, w//2
        y_forward_smallscale4_q = diff_fea_forward_q
        y_backward_smallscale4_q = diff_fea_backward_q
        y_forward_smallscale2_q = self.bn3_smallscale2(self.conv3_smallscale2(y_forward_smallscale2_q))
        y_backward_smallscale2_q = self.bn3_smallscale2(self.conv3_smallscale2(y_backward_smallscale2_q))
        y_forward_smallscale4_q = self.bn3_smallscale4(self.conv3_smallscale4(y_forward_smallscale4_q))
        y_backward_smallscale4_q = self.bn3_smallscale4(self.conv3_smallscale4(y_backward_smallscale4_q))
        y_forward_smallscale2_q = F.interpolate(y_forward_smallscale2_q, diff_fea_forward_q.size()[2:])
        y_backward_smallscale2_q = F.interpolate(y_backward_smallscale2_q, diff_fea_backward_q.size()[2:])
        # (20*8, 256, 7, 7)
        y_forward_q = self.conv3(1.0 / 3.0 * diff_fea_forward_q + 1.0 / 3.0 * y_forward_smallscale2_q + 1.0 / 3.0 * y_forward_smallscale4_q)
        y_backward_q = self.conv3(1.0 / 3.0 * diff_fea_backward_q + 1.0 / 3.0 * y_backward_smallscale2_q + 1.0 / 3.0 * y_backward_smallscale4_q)

        # RESHAPE: (20*8, 256, 49)->(20*8, 49, 256)->(20, 8, 49, 256)
        y_forward_s = y_forward_s.reshape(y_forward_s.size()[:-2] + (-1,)).permute(0, 2, 1).reshape(-1, self.n_segment, self.num_patches, self.channel // self.reduction)
        y_forward_q = y_forward_q.reshape(y_forward_q.size()[:-2] + (-1,)).permute(0, 2, 1).reshape(-1, self.n_segment, self.num_patches, self.channel // self.reduction)
        y_backward_s = y_backward_s.reshape(y_backward_s.size()[:-2] + (-1,)).permute(0, 2, 1).reshape(-1, self.n_segment, self.num_patches, self.channel // self.reduction)
        y_backward_q = y_backward_q.reshape(y_backward_q.size()[:-2] + (-1,)).permute(0, 2, 1).reshape(-1, self.n_segment, self.num_patches, self.channel // self.reduction)
        y_forward_s = self.norm_qk(y_forward_s)
        y_forward_q = self.norm_qk(y_forward_q)
        y_backward_q = self.norm_qk(y_backward_q)
        y_backward_s = self.norm_qk(y_backward_s)
        # forward s-q spatial attention
        # (25, 8, 49, 256)->(8, 25, 49, 256)->(8, 25*49, 256)
        y_forward_s = y_forward_s.permute(1, 0, 2, 3).reshape(self.n_segment, -1, self.channel // self.reduction).unsqueeze(0).repeat(query_num, 1, 1, 1)
        y_backward_s = y_backward_s.permute(1, 0, 2, 3).reshape(self.n_segment, -1, self.channel // self.reduction).unsqueeze(0).repeat(query_num, 1, 1, 1)
        y_forward_s = y_forward_s.reshape((-1,) + y_forward_s.size()[-2:])
        y_forward_q = y_forward_q.reshape((-1,) + y_forward_q.size()[-2:])
        y_backward_s = y_backward_s.reshape((-1,) + y_backward_s.size()[-2:])
        y_backward_q = y_backward_q.reshape((-1,) + y_backward_q.size()[-2:])
        attn1 = torch.bmm(y_forward_q, y_forward_s.transpose(-2, -1))
        attn1 = attn1 / np.power(self.temperature, 0.5)
        attn1 = self.softmax(attn1)
        attn1 = self.dropout(attn1)

        # backward s-q spatial attention
        attn2 = torch.bmm(y_backward_q, y_backward_s.transpose(-2, -1))
        attn2 = attn2 / np.power(self.temperature, 0.5)
        attn2 = self.softmax(attn2)
        attn2 = self.dropout(attn2)

        # (batch*8, 2048, 7, 7)->(20, 8, 2048, 7, 7) .reshape(query.size()[:-2] + (-1,)).permute(0, 2, 1)
        conv_query = self.conv(query).reshape(query_num, self.n_segment, self.channel, 7, 7)
        # (8, 25, 49, 2048)
        conv_support = self.conv(support).reshape(support_num, self.n_segment, self.channel, 7, 7)
        t_support = conv_support.reshape(support_num, self.n_segment, self.channel, self.num_patches).permute(1, 0, 3, 2)
        t_support = t_support.reshape(self.n_segment, -1, self.channel).unsqueeze(0).repeat(query_num, 1, 1, 1)
        t_support = t_support.reshape((-1,) + t_support.size()[-2:])
        # (20*8, 49, 2048)
        cross_q = 0.5 * torch.bmm(attn1, t_support) + 0.5 * torch.bmm(attn2, t_support)
        cross_q = cross_q.permute(0, 2, 1).reshape(query_num, self.n_segment, self.channel, 7, 7)
        query = query.reshape(query_num, self.n_segment, self.channel, 7, 7)
        support = support.reshape(support_num, self.n_segment, self.channel, 7, 7)
        query = query + self.gamma1 * conv_query + self.gamma2 * cross_q
        support = support + self.gamma1 * conv_support
        return support, query


class spa_cross_transfomer(nn.Module):
    def __init__(self, args, num_patches=49):
        super(spa_cross_transfomer, self).__init__()
        self.args = args
        self.num_patches = num_patches
        max_len = int(num_patches * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, 0.1, max_len)
        self.qk_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)

        self.norm_qk = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.class_softmax = torch.nn.Softmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, support_set, support_labels, queries):
        # support/queries: (batch, 8, 2048, 4, 4)
        n_queries = queries.shape[0]  # 20
        n_support = support_set.shape[0]  # 25
        # (25, 8, 2048, 16)->(25, 8, 16, 2048)->(25*8, 16, 2048)
        support_set = support_set.reshape(n_support, self.args.seq_len, self.args.trans_linear_in_dim, self.num_patches).\
            permute(0, 1, 3, 2).reshape(-1, self.num_patches, self.args.trans_linear_in_dim)
        queries = queries.reshape(n_queries, self.args.seq_len, self.args.trans_linear_in_dim, self.num_patches).\
            permute(0, 1, 3, 2).reshape(-1, self.num_patches, self.args.trans_linear_in_dim)
        support_set = self.pe(support_set)
        queries = self.pe(queries)
        support_set_ks = self.qk_linear(support_set)  # 25*8 x 49 x 1152
        queries_ks = self.qk_linear(queries)  # 20*8 x 49 x 1152
        support_set_vs = self.v_linear(support_set)  # 25*8 x 49 x 1152
        queries_vs = self.v_linear(queries)  # 20*8 x 49 x 1152

        mh_support_set_ks = self.norm_qk(support_set_ks).reshape(n_support, self.args.seq_len, self.num_patches, self.args.trans_linear_out_dim)\
            .to(device)  # 25 8 x 49 x 1152
        mh_queries_ks = self.norm_qk(queries_ks).reshape(n_queries, self.args.seq_len, self.num_patches, self.args.trans_linear_out_dim)\
            .to(device)  # 20 8 x 49 x 1152
        support_labels = support_labels.to(device)
        mh_support_set_vs = support_set_vs.reshape(n_support, self.args.seq_len, self.num_patches, self.args.trans_linear_out_dim)\
            .to(device)  # 25 8 x 49 x 1152
        mh_queries_vs = queries_vs.reshape(n_queries, self.args.seq_len, self.num_patches, self.args.trans_linear_out_dim)\
            .reshape(n_queries, -1, self.args.trans_linear_out_dim).to(device)  # 20 8 x 49 x 1152

        unique_labels = torch.unique(support_labels)  # 5
        all_distances_tensor = torch.zeros(n_queries, self.args.way)  # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))  # (5,8,49,1152)
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))  # (5,8,49,1152)
            k_bs = class_k.shape[0] # 5
            # (8, 5, 49, 1152)->(8, 5*49, 1152)
            class_k = class_k.permute(1, 0, 2, 3).reshape(self.args.seq_len, k_bs*self.num_patches, self.args.trans_linear_out_dim)
            class_v = class_v.permute(1, 0, 2, 3).reshape(self.args.seq_len, k_bs*self.num_patches, self.args.trans_linear_out_dim)

            # (20, 8, 49, 2048) (1, 8, 2048, 90)->(20, 8, 16, 90)
            class_scores = torch.matmul(mh_queries_ks, class_k.unsqueeze(0).transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_in_dim)
            class_scores = self.class_softmax(class_scores)
            class_task_support = torch.matmul(class_scores, class_v.unsqueeze(0)).reshape(n_queries, -1, self.args.trans_linear_out_dim) # 20 x 8 x 16 x 1152
            diff = mh_queries_vs - class_task_support  # 20 x 8 * 49 x 1152
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2  # 20
            distance = torch.div(norm_sq, self.args.seq_len ** 2 * self.num_patches)
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance  # 20
        return_dict = {'logits': all_distances_tensor}

        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class TemporalCrossTransformer(nn.Module):
    """
    Calculates the TRX distances for sequences in one direction (e.g. query to support).
    """
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)

        self.class_softmax = torch.nn.Softmax(dim=1)

        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples)  # 28

    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0]  # 20
        n_support = support_set.shape[0]  # 25

        # static pe after adding the position embedding
        support_set = self.pe(support_set)  # Support set is of shape 25 x 8 x 2048 -> 25 x 8 x 2048
        queries = self.pe(queries)  # Queries is of shape 20 x 8 x 2048 -> 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2)  # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2)  # 20 x 28 x 4096

        # apply linear maps for performing self-normalization in the next step and the key map's output
        '''
            support_set_ks is of shape 25 x 28 x 1152, where 1152 is the dimension of the key = query head. converting the 5-way*5-shot x 28(tuples).
            query_set_ks is of shape 20 x 28 x 1152 covering 4 query/sample*5-way x 28(number of tuples)
        '''
        support_set_ks = self.k_linear(support_set)  # 25 x 28 x 1152
        queries_ks = self.k_linear(queries)  # 20 x 28 x 1152
        support_set_vs = self.v_linear(support_set)  # 25 x 28 x 1152
        queries_vs = self.v_linear(queries)  # 20 x 28 x 1152

        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks).to(device)  # 25 x 28 x 1152
        mh_queries_ks = self.norm_k(queries_ks).to(device)  # 20 x 28 x 1152
        support_labels = support_labels.to(device)
        mh_support_set_vs = support_set_vs.to(device)  # 25 x 28 x 1152
        mh_queries_vs = queries_vs.to(device)  # 20 x 28 x 1152

        unique_labels = torch.unique(support_labels)  # 5

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        all_distances_tensor = torch.zeros(n_queries, self.args.way)  # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0,
                                         self._extract_class_indices(support_labels, c))  # 5 x 28 x 1152
            class_v = torch.index_select(mh_support_set_vs, 0,
                                         self._extract_class_indices(support_labels, c))  # 5 x 28 x 1152
            k_bs = class_k.shape[0]  # 5

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)  # 20 x 5 x 28 x 28

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0, 2, 1, 3)  # 20 x 28 x 5 x 28

            # [For the 20 queries' 28 tuple pairs, find the best match against the 5 selected support samples from the same class
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)  # 20 x 28 x 140
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]  # list(20) x 28 x 140
            class_scores = torch.cat(class_scores)  # 560 x 140 - concatenate all the scores for the tuples
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)  # 20 x 28 x 5 x 28
            class_scores = class_scores.permute(0, 2, 1, 3)  # 20 x 5 x 28 x 28

            # get query specific class prototype
            query_prototype = torch.matmul(class_scores, class_v)  # 20 x 5 x 28 x 1152
            query_prototype = torch.sum(query_prototype, dim=1).to(
                device)  # 20 x 28 x 1152 -> Sum across all the support set values of the corres. class

            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype  # 20 x 28 x 1152
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2  # 20
            distance = torch.div(norm_sq, self.tuples_len)  # 20

            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance  # 20

        return_dict = {'logits': all_distances_tensor}

        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class DMAGL(nn.Module):
    def __init__(self, args):
        super(DMAGL, self).__init__()

        self.train()
        self.args = args

        # Using ResNet Backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.Motion_Excitation = mSEModule(self.args.trans_linear_in_dim, n_segment=self.args.seq_len)
        # Temporal Cross Transformer for modelling temporal relations
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_mix1 = MLP_Mix_1(self.args.trans_linear_in_dim, n_segment=self.args.seq_len)
        # self.spa_trans = spa_cross_transfomer(args, num_patches=49)
        self.spa_mix2 = spa_mix_2(self.args.trans_linear_in_dim, n_segment=self.args.seq_len, num_patches=49)
        # self.ctrx = DistanceLoss(self.args, temporal_set_size=2, channel=self.args.trans_linear_in_dim) 暂时不用该模块

    def forward(self, context_images, context_labels, target_images):
        context_features = self.resnet(context_images)  # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_images)  # 160 x 2048 x 7 x 7

        context_features = self.Motion_Excitation(context_features)  # 200 x 2048 x 7 x 7
        target_features = self.Motion_Excitation(target_features)  # 160 x 2048 x 7 x 7

        context_features = self.mlp_mix1(context_features)  # 25 x 8 x 2048 x 7 x 7
        target_features = self.mlp_mix1(target_features)  # 20 x 8 x 2048 x 7 x 7

        # (5, 20, 8, 16, 2048)  (20, 8, 16, 2048)
        context_features = context_features.reshape(-1, self.args.trans_linear_in_dim, 7, 7)
        target_features = target_features.reshape(-1, self.args.trans_linear_in_dim, 7, 7)
        context_features, target_features = self.spa_mix2(context_features, target_features)

        context_features = self.avg_pool(context_features).squeeze()
        target_features = self.avg_pool(target_features).squeeze()

        unique_labels = torch.unique(context_labels)
        n_queries = target_features.shape[0]  # 20
        n_support = context_features.shape[0]  # 25
        context_features = rearrange(context_features, 'b s d -> (b s) d')  # [200, 2048]
        target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200, 2048]

        frame_sim = cos_sim(target_features, context_features)
        frame_dists = 1 - frame_sim
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)

        # calculate query -> support and support -> query
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(context_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        total_logist = - class_dists

        return_dict = {'logits': split_first_dim_linear(total_logist, [NUM_SAMPLES, n_queries])}

        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])