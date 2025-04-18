#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/11/8 12:55
# @Author  : lqh
# @python-version 3.10
# @File    : MyThermalRRModel.py
# @Software: PyCharm
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

import numpy as np
import os
import imageio
import time
import warnings
import sys
import copy
import json
from PIL import Image
import math


class MConfig:
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    MODEL_FILE_NAME = 'ThermRNet'
    DO_CHUNK = True
    CHUNK_LENGTH = 160  # 一个固定的帧长
    RESIZE_H = 96
    RESIZE_W = 96
    LR = 1e-4  # 训练集权重
    TOOLBOX_MODE = "train_and_test"
    TEST_USE_LAST_EPOCH = True
    BEGIN = 0.0
    END = 1.0
    DATA_FORMAT = "NCDHW"  # N:batch_size D: frame_size，C:channel，H:height，W:width
    PATCH_SIZE = 4
    DIM = 96
    FF_DIM = 144
    NUM_HEADS = 4
    NUM_LAYERS = 4
    THETA = 0.7
    DROP_RATE = 0.2
    GRA_SHARP = 2.0


config = MConfig()


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''


class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""

    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()

        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )

        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization

    def forward(self, x, gra_sharp):  # [B, 3*3*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        [B, P, C] = x.shape
        x = x.transpose(1, 2).view(B, C, P // 9, 3, 3)  # [B, dim, 40, 3, 3]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 3*3*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 3*3*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 3*3*40, dim]

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp

        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores


class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )

        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )

        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):  # [B, 3*3*40, 128]
        [B, P, C] = x.shape
        x = x.transpose(1, 2).view(B, C, P // 9, 3, 3)  # [B, dim, 40, 3, 3]
        x = self.fc1(x)  # x [B, ff_dim, 40, 3, 3]
        x = self.STConv(x)  # x [B, ff_dim, 40, 3, 3]
        x = self.fc2(x)  # x [B, dim, 40, 3, 3]
        x = x.flatten(2).transpose(1, 2)  # [B, 3*3*40, dim]

        return x

class Block_ST_TDC_gra_sharp(nn.Module):
    """
    变换器块Transformer Block
    包括多头自注意力机制、前馈神经网络以及相应的规范化和残差连接操作
    """

    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        # 计算输入特征的注意力权重，并根据这些权重对输入进行加权求和。通过多头机制，模型可以在不同的表示子空间中捕捉信息，从而增强其表达能力。
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        # 在自注意力机制和前馈网络之前应用层归一化（Layer Normalization），以稳定训练过程并加速收敛
        # BatchNorm 的目的是通过对每一批次（batch）数据进行标准化来减少内部协变量偏移。BatchNorm 的表现依赖于批量大小，较小的批量可能会导致不稳定的结果。
        # LayerNorm 的目标也是对输入进行标准化，但它是在单个样本的层面上进行的，而不是在整个批次上。LayerNorm 计算单个样本在给定层内的均值和方差，并使用这些统计量来标准化输入。
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        # 前馈网络，用于对每个位置的特征进行非线性变换。它由两个全连接层组成，中间夹着一个激活函数（通常是ReLU或ELU）。
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        # 通过残差连接将自注意力机制和前馈网络的输出与输入相加，并在连接之前应用丢弃层（Dropout）以防止过拟合。
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score


class Transformer_ST_TDC_gra_sharp(nn.Module):
    """
    Transformer with Self-Attentive Blocks
    一个基于变换器架构的模块，用于处理具有时空特征的数据，如视频序列。
    这个类封装了一系列变换器块（Block_ST_TDC_gra_sharp），并通过自注意力机制来捕捉输入数据中的长距离依赖关系
    自注意力机制允许模型关注输入序列中的不同部分，并根据它们之间的关系进行加权平均。这对于捕捉视频中的时空特征特别有用，因为视频中的动作或事件往往不是孤立发生的，而是与其他部分有关联
    多头注意力可以从不同的表示子空间中捕捉信息
    """

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        """
        num_layers：变换器模块中包含的变换器块的数量。
        dim：每个变换器块的输入和输出维度。
        num_heads：每个变换器块中多头自注意力机制的头数。
        ff_dim：位置感知前馈网络（Position-wise Feed-Forward Network）的隐藏层维度。
        dropout：用于防止过拟合的丢弃概率。
        theta：控制 CDC 层中原始卷积和中心差分卷积的比例。
        """
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score


class ThermRNet(nn.Module):
    # b, 3, 160, 96, 96
    def __init__(
            self,
            dim: int = 768,
            dropout_rate: float = 0.2,
            in_channels: int = 3,
            frame: int = 160,
            image_size: Optional[int] = 96,
    ):
        global config
        super().__init__()
        self.dropout_rate = dropout_rate
        self.image_size = image_size
        self.frame = frame
        self.dim = dim
        self.gra_sharp = config.GRA_SHARP

        # input(b, c, t, h, w),  (b, 3, 160, 96, 96)

        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 16, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout_rate),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),  # kernel_size[0] = 1表示时间维度保持不变
        )  # (b, dim//16, 80, 48, 48)

        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 16, dim // 8, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout_rate),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
        )  # (b, dim//8, 40, 24, 24)
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 8, dim // 4, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout_rate),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )  # (b, dim//4, 40, 12, 12)
        self.Stem3 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout_rate),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )  # (b, dim//2, 40, 6, 6)

        self.Stem4 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout_rate),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )  # (b, dim, 40, 3, 3)
        self.pos_embedding = nn.Parameter(torch.randn(1, 40 * 3 * 3, dim))  # 位置编码就是一个向量
        # 拿CNN做embedding了
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=config.NUM_LAYERS, dim=dim,
                                                         num_heads=config.NUM_HEADS,
                                                         ff_dim=config.FF_DIM, dropout=self.dropout_rate,
                                                         theta=config.THETA)  # (b, 3 *3 * 40, dim)
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=config.NUM_LAYERS, dim=dim,
                                                         num_heads=config.NUM_HEADS,
                                                         ff_dim=config.FF_DIM, dropout=self.dropout_rate,
                                                         theta=config.THETA)  # (b, 3 *3 * 40, dim)
        # 在残差连接后添加
        self.post_res_norm = nn.InstanceNorm3d(dim, affine=True)

        # 解码阶段需要逐步恢复分辨率
        # nn.Upsample(scale_factor=(8,1,1)) 将特征图在时间维度上放大8倍，而在空间维度上保持不变。这意味着如果输入是一个T×H×W的张量，那么输出将会是8T×H×W
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
            nn.Dropout3d(self.dropout_rate),
        )  # (b, dim, 80, 3, 3)
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
            nn.Dropout3d(self.dropout_rate),
        )  # (b, dim // 2, 160, 3, 3)
        self.final_maxpool3d = nn.MaxPool3d((1, 3, 3), stride=(1, 3, 3))  # 输出为(b, dim // 2, 160, 1, 1)
        # 用conv1d代替最后的全连接层
        self.ConvBlockLast = nn.Conv1d(dim // 2, 2, 1, stride=1, padding=0)  # 最后输出为两个通道，表示属于每个类的概率

        # # conv1d效果不好，换为全连接层
        # self.classifier = nn.Sequential(

        # )

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)

    def forward(self, x):
        b, c, t, fh, fw = x.shape
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, 64, 160, 64, 64]
        x = self.Stem3(x)
        x_stem4 = self.Stem4(x)

        x = x_stem4.flatten(2).transpose(1, 2)  # [B, 3 *3 * 40, dim]
        x += self.pos_embedding  # 添加位置编码
        x, Score1 = self.transformer1(x, self.gra_sharp)  # (b, 3 *3 * 40, dim)
        x, Score2 = self.transformer2(x, self.gra_sharp)  # (b, 3 *3 * 40, dim)
        x = x.transpose(1, 2).view(b, self.dim, 40, 3, 3)  # [B, dim, 40, 3, 3]

        # 添加残差连接：将 Stem4 的输出与 Transformer 的输出相加
        x = x + x_stem4  # 确保维度一致
        x = self.post_res_norm(x)

        x = self.upsample(x)  # [B, dim, 80, 3, 3]
        x = self.upsample2(x)  # [B, dim, 160, 3, 3]
        # features_last = self.final_maxpool3d(features_last).squeeze(-1).squeeze(-1)  # 去掉后面两个1的维度
        # x = self.final_maxpool3d(x).squeeze(-1).squeeze(-1)  # 去掉后面两个1的维度

        x = torch.mean(x, 3)  # x [B, dim, 160, 3]
        x = torch.mean(x, 3)  # x [B, dim, 160]
        logits = self.ConvBlockLast(x)  # 输出为(B, 2, 160)
        return logits


def apply_transform_to_video(video, transform):
    """
    Apply the same transformation to every frame of the video.

    Args:
        video (np.ndarray): Video with shape (channels, T, Height, Width).
        transform (callable): Transform function that takes PIL image and returns transformed PIL image.

    Returns:
        torch.Tensor: Transformed video with shape (channels, T, Height, Width).
    """
    # Ensure the video is in the correct format (float32 and normalized between 0 and 255)
    assert isinstance(video, np.ndarray)

    # Convert video frames to PIL images
    pil_images = [to_pil_image(np.uint8(frame)) for frame in video]  # (T, C, H, W)

    # Apply the same transform to each frame
    transformed_pil_images = [transform(img) for img in pil_images]
    # Convert back to a single tensor
    transformed_video = torch.stack(transformed_pil_images).permute(1, 0, 2, 3)  # (C, T, H, W)
    return transformed_video


# def clean_segments(sequence, fps):
#     """
#     去除不符合要求的呼与吸的间隔
#     sequence：元素为0或1的数组， 0吸入， 1呼出
#     """
#     lower_threshold = fps // 2  # 最快的呼吸不超过60次/分钟,也就是呼吸频率不超过1Hz,平均来说呼和吸的频率不超过2Hz
#     invalid_index_list = sequence >= lower_threshold
#     cleaned_sequence = sequence[(sequence >= lower_threshold)]
#     return cleaned_sequence

def clean_segments(sequence, fps):
    """
    去除不符合要求的呼与吸的间隔
    sequence：元素为0或1的数组， 0吸入， 1呼出
    """
    if len(sequence) == 1:
        return sequence
    lower_thr = fps // 2  # 最快的呼吸不超过60次/分钟,也就是呼吸频率不超过1Hz,平均来说呼和吸的频率不超过2Hz
    cleaned_sequence = []
    i = 0
    n = len(sequence)
    while i < n:
        length = sequence[i]
        if length < lower_thr:
            # 处理“短于阈值”的段：
            if not cleaned_sequence:
                # 这是第一个段，合到后面去
                if i + 1 < n:
                    sequence[i + 1] += length
                # 不把它放入 cleaned
            else:
                # 合到前面那段
                cleaned_sequence[-1] += length
                # 同时，如果它后面还有一段，也把那段也合到前面
                if i + 1 < n:
                    cleaned_sequence[-1] += sequence[i + 1]
                    i += 1  # 跳过下一个段
        else:
            # 够长，留下来
            cleaned_sequence.append(length)
        i += 1
    return cleaned_sequence


def count_segments(sequence, fps=25):
    """
    sequence：元素为0或1的数组， 0吸入， 1呼出
    k: 不少于k个连续的才会被计数   减轻模型偶尔预测错误的问题

    return:
        breath_interval:  一维数组，记录每个连续段的0或1的数目
    """
    if sequence is None:  # 如果序列为None，则没有段
        return 0, 0
    sequence_diff = np.diff(sequence)
    start_end_index = np.nonzero(sequence_diff)[0]
    breath_interval = np.diff(start_end_index)
    cleaned_sequence = clean_segments(breath_interval, fps)
    return cleaned_sequence

def get_valid_segments(sequence, fps):
    # segment_counts = count_segments(sequence, fps)
    # valid_counts = segment_counts[1:-1]  # 当segment_counts的长度不大于2时，valid_counts都为空列表

    valid_counts = count_segments(sequence, fps)
    return valid_counts


def get_rr(sequence, fps):
    """
    适合测量时间段长的，计算rr，通过有多少个呼与吸的转变来计算rr(一般忽略sequence两端不完整的呼吸)
    """
    segment_counts = count_segments(sequence, fps)
    num_in_out = len(segment_counts)
    if segment_counts[0] < fps / 2:  # 人为定义一个合适的值，决定第一段是否算数
        num_in_out -= 1
    if len(segment_counts) > 1 and segment_counts[-1] < fps / 2:  # 人为定义一个合适的值，决定最后一段是否算数
        num_in_out -= 1
    rr = num_in_out * fps * 30 / len(sequence)
    return rr


def get_rr_with_estimate(sequence, fps):
    """
    适合测量时间段短的， 通过平均时间段内呼吸的平均时长，来推断rr.(考虑了sequence两端不完整的呼吸)
    """
    segment_counts = count_segments(sequence, fps)
    if len(segment_counts) < 2:
        return 0
    valid_counts = segment_counts if len(segment_counts) == 2 else segment_counts[1:-1] # 当segment_counts的长度不大于2时，segment_counts[1:-1]为空列表
    even_indexed_elements = valid_counts[0::2]  # 从索引0开始，步长为2  (0,2,4,6,8...)下标的元素值组成的数组
    odd_indexed_elements = valid_counts[1::2]  # 从索引1开始，步长为2   (1,3,5,7,...)下标的元素值组成的数组
    if len(even_indexed_elements) == 0 or len(odd_indexed_elements) == 0:
        return 1 * fps * 30 / sum(valid_counts)  # 只有一次呼或只有一次吸

    # 计算这些元素的平均值
    average_valid_even = np.mean(even_indexed_elements)
    average_valid_odd = np.mean(odd_indexed_elements)
    # even_indexed_elements = segment_counts[0::2]  # 从索引0开始，步长为2
    # odd_indexed_elements = segment_counts[1::2]  # 从索引1开始，步长为2
    # num_in_out = np.sum(even_indexed_elements) / average_valid_odd + np.sum(odd_indexed_elements) / average_valid_even  # 注意因为一个是舍弃了开头和结尾处的，所以这里相除的时候even和odd是反着来的
    # rr = num_in_out * fps * 30 / len(sequence)

    rr = fps * 60 / (average_valid_even+average_valid_odd)
    return rr

def get_ThermRNet_model(model_path, device=torch.device('cpu'), image_size=96, chunk_len=160):
    model = ThermRNet(dim=96, frame=chunk_len, image_size=image_size)
    checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.to(device)

    mean = [0.456, 0.456, 0.456]
    std = [0.224, 0.224, 0.224]
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # This will convert PIL image to Tensor and scale values to [0, 1]
        transforms.Normalize(mean=mean, std=std)  # 然后进行标准化
    ])
    model.eval()
    return model, transform