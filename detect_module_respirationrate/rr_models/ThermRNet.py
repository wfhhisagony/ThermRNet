#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2025/12/8 15:08
# @Author  : lqh
# @python-version 3.10
# @File    : ThermRNet.py
# @Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,transforms, models
import torchvision
import pandas as pd
import scipy

import math
import random
import time
import numpy as np
from typing import Optional
import os
import imageio
import time
import warnings
import sys
import copy
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class SpectralGatingExpansion(nn.Module):
    def __init__(self, dim, expansion_factor=2):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        # complex weights stored as real/imag pairs, keep as float32 param
        self.complex_weight = nn.Parameter(torch.randn(hidden_dim, 1, 2, dtype=torch.float32) * 0.02)
        self.project_out = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        # x shape: (B*H*W, D, T) or (B, D, T)
        # Preserve original dtype and ensure FFT / complex ops run in full precision
        orig_dtype = x.dtype
        B, D, T = x.shape
        x = self.project_in(x)
        x = self.act(x)
        # Disable autocast (mixed-precision) for FFT and complex multiplication to avoid cuFFT half-precision limitations
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x.float()
            x_fft = torch.fft.rfft(x_fp32, n=T, dim=-1, norm='ortho')
            weight = torch.view_as_complex(self.complex_weight).to(x_fft.dtype)
            x_fft = x_fft * weight
            x_ifft = torch.fft.irfft(x_fft, n=T, dim=-1, norm='ortho')
            x = x_ifft.to(orig_dtype)
        x = self.project_out(x)
        return x


class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim, expansion_factor=2):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )
        self.temporal_spectral = SpectralGatingExpansion(dim, expansion_factor)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Input: (B, D, T, H, W)
        B, D, T, H, W = x.shape
        x_s = x.permute(0, 2, 1, 3, 4).reshape(B * T, D, H, W)
        x_s = self.spatial_conv(x_s)
        x_s = x_s.view(B, T, D, H, W).permute(0, 2, 1, 3, 4)
        x = x + x_s
        x_t = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, T)
        x_t = self.temporal_spectral(x_t)
        x_t = x_t.view(B, H, W, D, T).permute(0, 3, 4, 1, 2)
        x = x + x_t
        return x

class ThermRNet(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_dim, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.BatchNorm3d(base_dim),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.stage1 = nn.Sequential(
            SpatioTemporalBlock(base_dim),
            SpatioTemporalBlock(base_dim)
        )
        self.downsample = nn.Conv3d(base_dim, base_dim * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.stage2 = nn.Sequential(
            SpatioTemporalBlock(base_dim * 2),
            SpatioTemporalBlock(base_dim * 2),
            SpatioTemporalBlock(base_dim * 2)
        )
        self.final_dim = base_dim * 2
        self.classifier = nn.Sequential(
            nn.Conv1d(self.final_dim, self.final_dim, 1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv1d(self.final_dim, num_classes, 1)
        )

    def forward(self, x):
        # x: (B, 3, T, 72, 72)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x_pool = torch.mean(x, dim=(3, 4))  # (B, final_dim, T)
        logits = self.classifier(x_pool)   # (B, num_classes, T)
        return logits


def calculate_respiration_rate(predictions, fps, smooth_window_sec=0.4):
    """
    Calculates breathing rate from a binary sequence of frame classifications.

    Args:
        predictions (list or np.array): Binary sequence (0s and 1s) or probabilities.
        fps (float): Frames Per Second of the camera.
        smooth_window_sec (float): Window size for smoothing (default 0.5s is good for breathing).

    Returns:
        bpm (float): Breaths Per Minute.
        info (dict): Debug info containing smoothed signal and edge indices.
    """
    # 1. Convert to Numpy
    binary_signal = np.array(predictions)

    # 2. SMOOTHING (Median Filter)
    # Breathing is slow. We want to remove 'blips' shorter than smooth_window_sec.
    # Kernel size must be an odd integer.
    kernel_size = int(fps * smooth_window_sec)
    if kernel_size % 2 == 0: kernel_size += 1

    # Apply Median Filter: replaces each pixel with the median of its neighbors
    # This removes isolated noise (0001000) while preserving edges (0001111).
    smoothed_signal = scipy.signal.medfilt(binary_signal, kernel_size=kernel_size)

    # 3. EDGE DETECTION (Find transitions)
    # diff will be 1 at rising edge (0->1), -1 at falling edge (1->0), 0 otherwise
    diff_signal = np.diff(smoothed_signal)

    # Get indices where value is 1 (Start of Inhalation)
    rising_edges = np.where(diff_signal == 1)[0]

    # 4. CALCULATE RATE
    num_cycles = len(rising_edges)
    total_frames = len(predictions)
    total_duration_sec = total_frames / fps

    # Method A: Simple Count (Good for long videos, >1 minute)
    bpm_count = (num_cycles / total_duration_sec) * 60

    # Method B: Inter-Beat Interval (Better for short videos, <30 seconds)
    # We calculate the average time distance between rising edges.
    if num_cycles > 1:
        # Calculate distance between edges in frames
        intervals_frames = np.diff(rising_edges)
        avg_interval_frames = np.mean(intervals_frames)
        avg_interval_sec = avg_interval_frames / fps
        bpm_interval = 60.0 / avg_interval_sec
    else:
        # Fallback if we don't have two full breaths yet
        bpm_interval = bpm_count

    return bpm_interval, {
        "smoothed_signal": smoothed_signal,
        "rising_edges": rising_edges,
        "raw_signal": binary_signal
    }



def get_ThermRNet():
    pass