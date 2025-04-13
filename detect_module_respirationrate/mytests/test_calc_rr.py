#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2025/1/25 14:42
# @Author  : lqh
# @python-version 3.10
# @File    : test_calc_rr.py
# @Software: PyCharm
"""

import numpy as np

respiration_sequence = "00000000000000011111111111111111111000000000000000000011111111111111100000000000111111111111111000001111111111111111111000111111111011111111110101010000000001111111111111111111"
respiration_sequence = list(map(int, list(respiration_sequence)))

fps = 25


def get_breath_interval(sequence):
    sequence_diff = np.diff(sequence)
    breath_start_end_index = np.nonzero(sequence_diff)[0]
    breath_interval = np.diff(breath_start_end_index)
    return breath_interval

def clean_sequence(sequence, fps):
    lower_threshold = fps // 2# 最快的呼吸不超过60次/分钟,也就是呼吸频率不超过1Hz,平均来说呼和吸的频率不超过2Hz
    # cleaned_rr = sequence[(sequence >= lower_threshold)]

    cleaned_rr = sequence.copy().astype(float)
    modified = True

    while modified:
        modified = False
        # 找出所有需要处理的候选位置
        candidates = np.where(cleaned_rr < lower_threshold)[0]
        if len(candidates) == 0:
            break
        # 创建掩码标记待删除位置
        to_delete = np.zeros_like(cleaned_rr, dtype=bool)

        # 逆序处理避免索引错位
        for i in reversed(candidates):
            if i == 0 or to_delete[i]:
                continue  # 跳过首元素和已处理元素
            # 执行合并操作
            cleaned_rr[i - 1] += cleaned_rr[i]
            to_delete[i] = True
            modified = True
        # 生成新数组
        cleaned_rr = cleaned_rr[~to_delete]
    return cleaned_rr

def get_rr(sequence, fps):
    breath_interval = get_breath_interval(sequence)
    cleaned_sequence = clean_sequence(breath_interval, fps)
    if len(cleaned_sequence) == 0:  # 捕获的序列中没有检测到一次完整的呼吸
        return 0
    if len(cleaned_sequence) > 1 and len(cleaned_sequence) % 2 != 0:
        cleaned_sequence = cleaned_sequence[1:]
    mean_breath_interval = np.mean(cleaned_sequence)
    rr = fps * 30 / mean_breath_interval  # *60 / 2 = *30 因为呼和吸构成一次完整的呼吸
    return rr

if __name__ == '__main__':
    rr = get_rr(respiration_sequence, fps)
    print(rr)
