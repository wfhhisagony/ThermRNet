#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/12/3 14:12
# @Author  : lqh
# @python-version 3.10
# @File    : post_process.py
# @Software: PyCharm
"""
import numpy as np
import scipy
import scipy.io
import scipy.signal
from scipy.signal import butter, resample, welch, detrend, savgol_filter, find_peaks
from scipy.sparse import spdiags


def get_rr_with_fft(y, sr=25, min=5, max=45):  # 建议y的信号长度不少于12秒(len(y) >= sr * 12)
    """
    @param:
        y: filtered rr signal
        sr: signal rate
        min: low frequency(in minute) to be removed
        max: high frequency(in minute) to be removed
    @return:
        main frequency(in minute) in y
    """
    p, q = welch(y, sr, nfft=1e6 / sr, nperseg=np.min((len(y) - 1, 512)))
    return p[(p > min / 60) & (p < max / 60)][np.argmax(q[(p > min / 60) & (p < max / 60)])] * 60


def get_psd(y, sr=25, min=5, max=45):
    p, q = welch(y, sr, nfft=1e6 / sr, nperseg=np.min((len(y) - 1, 512)))
    return q[(p > min / 60) & (p < max / 60)]

def get_rr_with_peak_distance(y, sr=25):
    """
    该方法适用于y的信号质量较好时、噪声少时
    @param:
        y: filtered rr signal
        sr: signal rate
    @return:
        rr: respiration rate(bpm)
    """
    y_std = np.std(y)
    prominence = max(y_std * 0.2, 1.3)
    # 60 / 45 = 1.33 > 1
    peaks, _ = find_peaks(y, distance=int(1 * sr), prominence=prominence)
    if len(peaks) <= 1:
        return 60 * sr * len(peaks) / len(y)
    # 计算平均距离
    average_distance = np.mean(np.diff(peaks))
    rr = 60 * sr / average_distance
    return rr

def get_rr_with_peak_number(y, sr=25):
    """
    该方法在y的信号长度较长时适用(建议y的信号长度为25秒以上)，适用于非平稳信号的指标检测
    @param:
        y: filtered rr signal
        sr: signal rate
    @return:
        rr: respiration rate(bpm)
    """
    y_std = np.std(y)
    prominence = max(y_std * 0.2, 1.3)
    # 60 / 45 = 1.33 > 1
    peaks, _ = find_peaks(y, distance=int(1 * sr), prominence=prominence)
    rr = 60 * sr * len(peaks) / len(y)
    return rr

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):  # 基于平滑先验的去趋势处理
    """Detrend signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)


# 示性函数
def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0


def normalize(x):
    return (x - x.mean()) / x.std()

def custom_convolve(signal, window_size):  # np.convolve mode=='same'时的一个替代方案，不像'same'模式一样填充0，而是原封不动地保留窗口长度不足以convolve的信号点
    # 确保窗口大小是奇数，以便有一个中心点
    if window_size % 2 == 0:
        raise ValueError("Window size should be an odd number for a clear center point.")

    half_window = window_size // 2
    smoothed_signal = np.zeros_like(signal)

    for i in range(len(signal)):
        # 计算当前点的有效窗口
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        # 如果窗口内的点少于window_size，则使用实际可用的点
        current_window = signal[start:end]
        if len(current_window) > 0:
            smoothed_signal[i] = np.mean(current_window)
        else:
            # 如果没有有效的点，可以设置为原始值或其他处理方式
            smoothed_signal[i] = signal[i]
    return smoothed_signal


def filt_rrSignal_bandpassFilter(signal, sr, LP, HP):
    """
    @param:
        signal: one dimension list
        sr: signal rate
    @return:
        smoothed_signal
    """
    smoothed_signal = detrend(signal)
    # smoothed_signal = signal
    [b, a] = butter(1, [LP / sr * 2, HP / sr * 2], btype='bandpass')  # 制作带通滤波器
    smoothed_signal = scipy.signal.filtfilt(b, a, np.double(smoothed_signal))
    return smoothed_signal


def filt_rrSignal_savgolFilter(signal, sr, convolve_mode='valid'):
    """
    @param:
        signal: one dimension list
        sr: signal rate
    @return:
        smoothed_signal
    """
    window_size = int(sr * 0.8)  # 0.8 second window
    # smoothed_signal = _detrend(np.array(signal), 100)
    smoothed_signal = detrend(signal)  # scipy.signal.detrend方法处理速度更快，效果略和_detrend有些不同
    # 移动平均过滤
    # if convolve_mode == 'same':
    #     smoothed_signal = custom_convolve(smoothed_signal, window_size//2*2+1)
    smoothed_signal = np.convolve(smoothed_signal, np.ones(window_size) / window_size, mode=convolve_mode)  # valid模式会使得数组长度变短，same模式保留原长度，但效果通常不如valid
    # 60/45 = 1.33 > 1.2
    window_length = min(int(len(signal) / 2) * 2 - 1, int(sr * 1.2) + int(
            sr * 1.2) % 2 + 1)  # window_length必须为奇数且不能超过len(x)。它越大，则平滑效果越明显；越小，则更贴近原始曲线。
    polyorder = 3  # 多项式阶数  它越小，则平滑效果越明显；越大，则更贴近原始曲线。
    smoothed_signal = savgol_filter(smoothed_signal, window_length=window_length, polyorder=polyorder, mode='nearest')
    return smoothed_signal
