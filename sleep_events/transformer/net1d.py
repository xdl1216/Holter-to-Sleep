# net1d.py
"""
a modularized deep neural network for 1-d signal data, pytorch version
Modified for Time-Frequency Fusion and 2-Task Output (Arousal, Respiratory)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride, groups=groups)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        net = F.pad(x, (p // 2, p - p // 2), "constant", 0)
        return self.conv(net)

class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.max_pool = torch.nn.MaxPool1d(kernel_size=kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x):
        p = max(0, self.kernel_size - 1)
        net = F.pad(x, (p // 2, p - p // 2), "constant", 0)
        return self.max_pool(net)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups,
                 downsample, is_first_block=False, use_bn=True, use_do=True):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.middle_channels = int(self.out_channels * self.ratio)

        if self.is_first_block:
            self.bn1 = nn.Identity()
            self.do1 = nn.Identity()
        else:
            self.bn1 = nn.BatchNorm1d(in_channels)
            self.do1 = nn.Dropout(p=0.5)
        self.activation1 = Swish()
        self.conv1 = MyConv1dPadSame(self.in_channels, self.middle_channels, 1, 1, 1)

        self.bn2 = nn.BatchNorm1d(self.middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(self.middle_channels, self.middle_channels, self.kernel_size, self.stride, self.groups)

        self.bn3 = nn.BatchNorm1d(self.middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(self.middle_channels, self.out_channels, 1, 1, 1)

        r = 2
        self.se_fc1 = nn.Linear(self.out_channels, self.out_channels // r)
        self.se_fc2 = nn.Linear(self.out_channels // r, self.out_channels)
        self.se_activation = Swish()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x
        out = x
        if self.use_bn: out = self.bn1(out)
        out = self.activation1(out)
        if self.use_do: out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn: out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do: out = self.do2(out)
        out = self.conv2(out)

        if self.use_bn: out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do: out = self.do3(out)
        out = self.conv3(out)

        se = out.mean(-1)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = torch.sigmoid(se)
        out = torch.einsum('abc,ab->abc', out, se)

        if self.downsample: identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)
        out += identity
        return out

class BasicStage(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, i_stage, m_blocks, use_bn=True, use_do=True, verbose=False):
        super(BasicStage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_list = nn.ModuleList()
        for i_block in range(m_blocks):
            is_first = (i_stage == 0 and i_block == 0)
            downsample = (i_block == 0)
            current_stride = stride if downsample else 1
            tmp_in = in_channels if i_block == 0 else out_channels
            self.block_list.append(BasicBlock(tmp_in, out_channels, ratio, kernel_size, current_stride, groups, downsample, is_first, use_bn, use_do))

    def forward(self, x):
        out = x
        for block in self.block_list:
            out = block(out)
        return out

class Net1D(nn.Module):
    """
    base 1D CNN
    """
    def __init__(self, in_channels, base_filters, ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes, use_bn=True, use_do=True, verbose=False):
        super(Net1D, self).__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_stages = len(filter_list)
        self.use_bn = use_bn
        
        self.first_conv = MyConv1dPadSame(in_channels, base_filters, kernel_size, 2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        self.stage_list = nn.ModuleList()
        in_ch = base_filters
        for i in range(self.n_stages):
            out_ch = filter_list[i]
            self.stage_list.append(BasicStage(in_ch, out_ch, ratio, kernel_size, stride, out_ch//groups_width, i, m_blocks_list[i], use_bn, use_do, verbose))
            in_ch = out_ch
        
        self.dense = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        out = self.first_conv(x)
        if self.use_bn: out = self.first_bn(out)
        out = self.first_activation(out)
        for stage in self.stage_list:
            out = stage(out)
        out = out.mean(-1)
        out = self.dense(out)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, in_features, reduction=4):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.fc(x)
        return attn

class MultiTaskNet1D(Net1D):
    def __init__(self, in_channels, base_filters, ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes_list, use_bn=True, use_do=True, verbose=False):
        super(MultiTaskNet1D, self).__init__(in_channels, base_filters, ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes_list[0], use_bn, use_do, verbose)
        
        if hasattr(self, 'dense'):
            delattr(self, 'dense')

        time_feature_dim = filter_list[-1]

        self.freq_fc = nn.Sequential(
            nn.Linear(39, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.freq_attention = AttentionLayer(128)

        self.time_fc = nn.Sequential(
            nn.Linear(time_feature_dim, 1024),
            nn.ReLU()
        )
        self.time_attention = AttentionLayer(1024)

        fusion_dim = 1024 + 128
        
        self.head_arousal = nn.Linear(fusion_dim, n_classes_list[0])
        self.head_respiratory = nn.Linear(fusion_dim, n_classes_list[1])
        
    def forward(self, x, x_freq):
        # --- Time Domain Branch ---
        out = x
        out = self.first_conv(out)
        if self.use_bn: out = self.first_bn(out)
        out = self.first_activation(out)
        
        for i_stage in range(self.n_stages):
            out = self.stage_list[i_stage](out)
        
        # Global Average Pooling: (B, 1024, L) -> (B, 1024)
        out_time = out.mean(-1)
        
        # Time processing & Attention
        out_time = self.time_fc(out_time)        
        att_time = self.time_attention(out_time) 
        out_time = out_time * att_time

        # --- Frequency Domain Branch ---
        out_freq = self.freq_fc(x_freq)          
        att_freq = self.freq_attention(out_freq) 
        out_freq = out_freq * att_freq

        # --- Fusion ---
        out_fusion = torch.cat([out_time, out_freq], dim=1)

        # --- Multi-Task Heads ---
        out_arousal = self.head_arousal(out_fusion)
        out_respiratory = self.head_respiratory(out_fusion)
        
        return out_arousal, out_respiratory
