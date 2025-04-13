#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/12/02 15:28
# @Author  : lqh
# @python-version 3.10
# @Software: PyCharm
# @Description:
    通过向RRDetector中传入包含鼻孔的热成像视频片段来进行呼吸率识别

"""
import os
import sys
import logging

import cv2

import rr_detect_config

# 创建日志记录器
logger = logging.getLogger()
# 设置日志记录器的级别
logger.setLevel(rr_detect_config.LOGGING_LEVEL)
logging.basicConfig(level=rr_detect_config.LOGGING_LEVEL, format="%(levelname)s::%(funcName)s:  %(message)s",
                    stream=sys.stdout)

import torch

from signal_filters import post_process

import numpy as np
from ultralytics import YOLO

# THERMAL_FRAME_WIDTH = 256
# THERMAL_FRAME_HEIGHT = 192

# 定义一个窗口
# CV2_WINDOW_NAME = 'Cropped Nose Frame'
# cv2.namedWindow(CV2_WINDOW_NAME, cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口

class MyYoloPoseModel:
    def __init__(self):
        self.model_path = rr_detect_config.NOSEDETECTOR_MODEL_PATH
        self.model = YOLO(self.model_path, task="pose")  # pretrained YOLOv8n model
        self.testOne()  # 打通pipeline，减少后续预处理的时间

    def testOne(self):
        # self.model.predict(test_img_path, save=False, imgsz=640)  # return a list of Results objects
        self.model.predict(save=False, imgsz=640)  # return a list of Results objects

    def detect(self, np_img):
        """
        @param:
            np_img:  numpy数组存储到三通道热像图
        """
        # verbose = False关闭 控制台推理速度输出
        results = self.model.predict(np_img, save=False, imgsz=640, verbose=False)  # return a list of Results objects
        if len(results) > 0:
            result = results[0]
            if len(result.boxes) != 0:
                n_rect = np.array(result.boxes.xyxyn[0].cpu())  # (4,) 归一化的坐标
                n_face_w = result.boxes.xywhn[0][2].item()  # 归一化的脸宽
                n_face_h = result.boxes.xywhn[0][3].item()  # 归一化的脸高
                n_keypoints = np.array(result.keypoints.xyn[0].cpu())  # (5,2)  (x,y)
                # 展示YOLO模型特征点的位置
                # xmin, ymin, xmax, ymax = [item.int().item() for item in result.boxes.xyxy[0]]  # tensor转换为python类型的普通int
                # cv2.rectangle(np_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                # xys = keypoints.xy[0].int()
                # keypoints = []
                # for (x, y) in xys:
                #     keypoints.append((x.item(), y.item()))
                #     # cv2.circle(np_img, (x.item(), y.item()), 1, (0, 255, 0), 4)
                return n_rect, n_keypoints, n_face_w, n_face_h
        return (), [], 0, 0

    def detect_batch(self, frames):
        # frames_copy = np.transpose(frames, (0, 3, 1, 2))    # 通过tensor批量传入
        frames_tensor = (torch.Tensor(frames) / 255.0).permute(0, 3, 1, 2)   # 通过tensor批量传入
        results = self.model.predict(frames_tensor, stream=True, save=False, imgsz=640, verbose=False)  # return a list of Results objects
        # ans = []
        for r in results:
            if len(r) > 0:
                result = r[0]
                if len(result.boxes) != 0:
                    n_rect = np.array(result.boxes.xyxyn[0].cpu())  # (4,) 归一化的坐标
                    n_face_w = result.boxes.xywhn[0][2].item()  # 归一化的脸宽
                    n_face_h = result.boxes.xywhn[0][3].item()  # 归一化的脸高
                    n_keypoints = np.array(result.keypoints.xyn[0].cpu())  # (5,2)
                    # ans.append([n_rect, n_keypoints, n_face_w, n_face_h])
                    yield n_rect, n_keypoints, n_face_w, n_face_h
            else:
                # ans.append([(), [], 0, 0])
                yield (), [], 0, 0
        # return ans

class NoseCropper:
    """
        利用YOLO实现对热成像视频片段中的鼻子区域截取
    """

    def __init__(self):

        self.face_detect_model = MyYoloPoseModel()
        # 上次检测的鼻子的位置
        self.last_row = 0
        self.last_col = 0
        self.last_height = 1
        self.last_width = 1
        self.face_size = [1, 1]  # width、height

    def get_crop_frame(self, src_frame):
        """
        @param:
            src_frame: numpy array, thermal video frame, the frame has 3 channels(you can use gray img copy to 3 channel img)
        @return:
            crop_frames: cropped nose frame
        """
        h, w, c = src_frame.shape
        n_rect, n_kpt, n_face_w, n_face_h = self.face_detect_model.detect(src_frame)
        if len(n_rect) != 0:
            self.face_size[0] = n_face_w * w
            self.face_size[1] = n_face_h * h
            half_nose_width, nose_height_top, nose_height_bottom = self.face_size[0] * 0.14, self.face_size[1] * 0.01, \
                                                                   self.face_size[1] * 0.16
            n_nose_kpt = n_kpt[2]
            nose_kpt = [n_nose_kpt[0] * w, n_nose_kpt[1] * h]
            # 更新这次检测的人脸位置
            self.last_row = np.clip(round(nose_kpt[1] - nose_height_top), 0, h)
            self.last_col = np.clip(round(nose_kpt[0] - half_nose_width), 0, w)
            self.last_width = np.clip(round(half_nose_width * 2), 0, w)
            self.last_height = np.clip(round(nose_height_top + nose_height_bottom), 0, h)

        if self.last_height == 0 or self.last_width == 0:  # 检测的结果非常失败，检测出了个空框
            self.last_height = self.last_width = 1  # 保证截取的至少不是空帧
        crop_frame = np.copy(src_frame[self.last_row:self.last_row + self.last_height,
                         self.last_col:self.last_col + self.last_width])
        return crop_frame

    def get_crop_frames(self, src_frames, face_detect_freq=25):
        """
        @param:
            src_frames: numpy array, thermal video frames, every frame has 3 channels(you can use gray img copy to 3 channel img)
            face_detect_freq:  face detect frequency
        @return:
            crop_frames: a list(python list) of cropped nose frames
        """
        crop_frames = []
        frame_num, h, w, c = src_frames.shape
        cnt = -1
        for src_frame in src_frames:
            cnt += 1
            if cnt % face_detect_freq == 0:
                crop_frame = self.get_crop_frame(src_frame)
            else:
                crop_frame = np.copy(src_frame[self.last_row:self.last_row + self.last_height,
                                     self.last_col:self.last_col + self.last_width])
            # 显示截取的图片
            # cv2.imshow(CV2_WINDOW_NAME, crop_frame)
            # cv2.waitKey(10)
            crop_frames.append(crop_frame)
        return crop_frames


class RRDetector:
    def __init__(self, face_detect_freq=25):
        self.face_detect_freq = face_detect_freq
        self.nose_cropper = NoseCropper()

    def get_crop_frames(self, src_frames):
        return self.nose_cropper.get_crop_frames(src_frames, self.face_detect_freq)

    def get_rr_list(self, src_frames, fps=25, rr_method=rr_detect_config.RRMethods.FFT,
                    signal_filter=rr_detect_config.FilterMethods.bandpass, window_size=25*12, stride=25*6):
        """
        @param:
            src_frames:  视频帧（存储在numpy中), 这里每个src_frame 都是 三通道格式的
            fps:  视频的帧率
        @return:
            rr_list:  每个窗口中检测出的rr值
        """

        crop_frames = self.get_crop_frames(src_frames)
        return self.get_rr_list_with_cropFrames(crop_frames, fps, rr_method=rr_method, signal_filter=signal_filter, window_size=window_size, stride=stride)

    def get_rr_list_with_cropFrames(self, crop_frames, fps=25, rr_method=rr_detect_config.RRMethods.FFT,
                                    signal_filter=rr_detect_config.FilterMethods.bandpass, window_size=25*12, stride=25*6):
        """
        @param:
            crop_frames:  已经过截取的视频帧（存储在numpy中), 这里每个crop_frame 都是  三通道格式的
            fps:  视频的帧率(采样率)
            rr_method:
            window_size:  每个窗口的采样点数, 窗口的时间长度detect_seconds = window_size / fps
            stride:  下一个窗口开始位置与上一个窗口开始位置间隔的采样点数目, 即 s1 + stride = s2
        @return:
            rr_list:  每个窗口中检测出的rr值
        """
        rr_list = []

        raw_signal = self.get_raw_signal_with_cropFrames(crop_frames, fps)
        smoothed_signal = self.get_smoothed_signal_with_rawSignal(raw_signal, fps, signal_filter)

        signal_length = len(smoothed_signal)
        window_increment = window_size - stride
        overlapped_window_num = (signal_length - window_size) // window_increment
        first_window_signal = smoothed_signal[:window_size]
        rr = self.get_rr(first_window_signal, fps, rr_method)
        rr_list.append(rr)
        if signal_length < window_size:
            logger.warning('in get_rr_list_with_cropFrames(): signal_length < window_size. use signal_length instead')
            return rr_list
        last_iloc = window_size
        for window_index in range(overlapped_window_num):
            last_iloc = last_iloc + window_increment
            window_signal = smoothed_signal[last_iloc - window_size:last_iloc]
            rr = self.get_rr(window_signal, fps, rr_method)
            rr_list.append(rr)
        if overlapped_window_num * window_increment < signal_length - window_size:  # 最后一批数据不足window_increment，则用前面的数据补足
            window_signal = smoothed_signal[-window_size:]
            rr = self.get_rr(window_signal, fps, rr_method)
            rr_list.append(rr)
        return rr_list

    def get_rr(self, smoothed_signal, fps, rr_method):
        if rr_method == rr_detect_config.RRMethods.FFT:
            rr = post_process.get_rr_with_fft(smoothed_signal, fps)
        elif rr_method == rr_detect_config.RRMethods.Peak:
            rr = post_process.get_rr_with_peak_distance(smoothed_signal, fps)
        else:
            raise Exception(f'No such rr_method: {rr_method}')
        return rr

    def get_raw_signal_with_cropFrames(self, crop_frames, fps):
        # TODO: 使用更好地提取raw呼吸信号的方法
        raw_signal = [np.mean(crop_frame) for crop_frame in crop_frames]  # 使用整张图片的像素平均值作为呼吸信号
        return raw_signal

    def get_smoothed_signal_with_rawSignal(self, raw_signal, fps=25, signal_filter=rr_detect_config.FilterMethods.bandpass):
        """
        @param:
            raw_signal: one dimension numpy array, raw rr signal
            fps:  sampling rate of the raw_signal
            signal_filter:  use which filter to smooth the raw_signal
        @return:
            smoothed_signal
        """
        if signal_filter == rr_detect_config.FilterMethods.bandpass:
            return post_process.filt_rrSignal_bandpassFilter(raw_signal, fps, 0.08, 0.75)
        elif signal_filter == rr_detect_config.FilterMethods.savgol:
            return post_process.filt_rrSignal_savgolFilter(raw_signal, fps)
        else:
            raise Exception(f'No Such filter:{signal_filter}')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fps = 25
    rr_method = rr_detect_config.RRMethods.Peak
    signal_filter = rr_detect_config.FilterMethods.savgol

    window_size = fps * 12
    # stride = window_size // 2
    stride = 1
    face_detect_freq = stride

    rr_detector = RRDetector(face_detect_freq=face_detect_freq)  # 让其在一个窗口内只检测一次人脸,避免人脸窗口抖动造成检测误差，但是这要求用户尽量保持头部不动

    vid_path = r'../test_data_files/test_data_rr/2.npy'

    src_frames = np.load(vid_path)
    rr_list = rr_detector.get_rr_list(src_frames, fps, rr_method, signal_filter, window_size, stride)
    print(f'mean rr: {np.mean(rr_list)}')

    # 绘图
    crop_frames = rr_detector.get_crop_frames(src_frames)
    raw_signal = rr_detector.get_raw_signal_with_cropFrames(crop_frames, fps)
    smoothed_signal = rr_detector.get_smoothed_signal_with_rawSignal(raw_signal, fps, signal_filter)
    def plot_data(data):
        # 绘制线图
        plt.plot(data, label='respiration signal')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Respiration Signal')
        plt.legend()
        plt.show()
    plot_data(smoothed_signal)

    # vid_dir = r'../test_data_files/test_data_rr'
    # vid_name_list = os.listdir(vid_dir)
    # vid_name_list = [vid_name for vid_name in vid_name_list if vid_name.endswith('npy')]
    # for vid_name in vid_name_list:
    #     print(vid_name)
    #     vid_path = os.path.join(vid_dir, vid_name)
    #     src_frames = np.load(vid_path)
    #     rr_list = rr_detector.get_rr_list(src_frames, fps, rr_method, signal_filter, window_size, stride)
    #     print(f'mean rr: {np.mean(rr_list)}')