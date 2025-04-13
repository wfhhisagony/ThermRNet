#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/12/2 21:27
# @Author  : lqh
# @python-version 3.10
# @File    : _config.py
# @Software: PyCharm
"""
import logging

LOGGING_LEVEL = logging.INFO
import os
# 获取当前文件的绝对路径
current_config_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_config_directory = os.path.dirname(current_config_file_path)
# 获取预训练模型所在的目录
PRETRAINED_MODEL_FIL_DIR = os.path.join(current_config_directory, 'pretrained_model_files')
# NOSEDETECTOR_MODEL_PATH = os.path.join(PRETRAINED_MODEL_FIL_DIR, 'yolov11n_tf.pt')
NOSEDETECTOR_MODEL_PATH = os.path.join(PRETRAINED_MODEL_FIL_DIR, 'yolov11s_tf.pt')
RRDETECTOR_MODEL_PATH = os.path.join(PRETRAINED_MODEL_FIL_DIR, 'ThermRNet.pth')

class RRMethods:
    Peak = 'Peak'
    FFT = 'FFT'
class FilterMethods:
    savgol = 'savgol'
    bandpass = 'bandpass'
# vars()获取类的所有属性和方法
RR_METHODS = [value for key, value in vars(RRMethods).items() if not key.startswith('__') and not key.endswith('__')]
# RR_METHODS = ['Peak', 'FFT']
