#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2025/1/11 15:41
# @Author  : lqh
# @python-version 3.10
# @File    : rr_detector_model.py
# @Software: PyCharm

使用 my3DCNN模型进行呼吸率估计
"""
import torch
import numpy as np
from rr_models import get_ThermRNet_model, apply_transform_to_video, get_rr_with_estimate, get_valid_segments
import cv2
from rr_detector import MyYoloPoseModel
from rr_detect_config import RRDETECTOR_MODEL_PATH

WIDTH = 192
HEIGHT = 256
THERMAL_FPS = 25

# # 定义一个窗口
# CV2_WINDOW_NAME = 'Cropped Nose Frame'
# cv2.namedWindow(CV2_WINDOW_NAME, cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口


def resize_raw_crop_frames(crop_frames, resize_size=96):  # BGR -> RGB
    resize_width = resize_height = resize_size
    frame_num = len(crop_frames)
    resized_clip_frames = np.zeros((frame_num, resize_height, resize_width, 3), dtype=np.float32)
    for i in range(frame_num):
        crop_frame = crop_frames[i]
        # crop_frame = cv2.cvtColor(crop_frames[i], cv2.COLOR_BGR2RGB)
        resized_clip_frames[i] = cv2.resize(crop_frame, (resize_width, resize_height),
                                            interpolation=cv2.INTER_AREA)
    return resized_clip_frames


class RRDetectorModel:
    def __init__(self, width=WIDTH, height=HEIGHT, fps=THERMAL_FPS):
        """
        @param
            width:  width of the thermal camera image
            height:  height of the thermal camera image
            fps:  sample rate of thermal camera
        """
        self.fps = fps
        self.image_size = 96
        self.chunk_len = 160
        self.device = torch.device('cuda:0')
        self.thermal_camera_width = width
        self.thermal_camera_height = height
        self.nose_frames = []

        self.model, self.transform = get_ThermRNet_model(RRDETECTOR_MODEL_PATH, self.device, self.image_size, self.chunk_len)
        self.face_detector = MyYoloPoseModel()
        self.cnt = 0
        self.face_size = [0, 0]
        self.kpt = [(0,0),( self.thermal_camera_width-1, 0), (self.thermal_camera_width//2, self.thermal_camera_height//2), (0, self.thermal_camera_height-1), (self.thermal_camera_width-1, self.thermal_camera_height-1)] # 左眼、右眼、鼻尖、左嘴角、右嘴角的xy坐标  初始化，防止第一帧检测失败的情况

    def get_nose_frame(self, bgr_img):
        if self.cnt == 0:
            img_h, img_w = bgr_img.shape[0], bgr_img.shape[1]
            self.thermal_camera_width = img_w
            self.thermal_camera_height = img_h
            n_rect, n_kpt, n_face_w, n_face_h = self.face_detector.detect(bgr_img)
            if len(n_rect) != 0:
                self.rect = (
                    (round(n_rect[0] * img_w), round(n_rect[1] * img_h)),
                    (round(n_rect[2] * img_w), round(n_rect[3] * img_h))
                )
                self.kpt = []
                self.face_size[0] = n_face_w * img_w
                self.face_size[1] = n_face_h * img_h
                for p in n_kpt:
                    self.kpt.append((round(p[0] * img_w), round(p[1] * img_h)))
        # 存入鼻子周围的热像图帧
        half_nose_width, nose_height_top, nose_height_bottom = self.face_size[0] * 0.14, self.face_size[1] * 0.01, self.face_size[1] * 0.16
        nose_start_x = np.clip(round(self.kpt[2][0] - half_nose_width), 0, self.thermal_camera_width)
        nose_start_y = np.clip(round(self.kpt[2][1] - nose_height_top), 0, self.thermal_camera_height)
        nose_end_x = np.clip(round(self.kpt[2][0] + half_nose_width), 0, self.thermal_camera_width)
        nose_end_y = np.clip(round(self.kpt[2][1] + nose_height_bottom), 0, self.thermal_camera_height)
        self.cnt = (self.cnt+1) % self.fps
        nose_frame = bgr_img[nose_start_y:nose_end_y, nose_start_x:nose_end_x]
        if 0 in nose_frame.shape: # 鼻子检测失败
            nose_frame = bgr_img
        # cv2.imshow(CV2_WINDOW_NAME, nose_frame)
        # cv2.waitKey(10)
        return nose_frame

    def get_nose_frames(self, src_frames):
        """
        param
            src_frames:  原始图像，未经过鼻部区域截取   不少于160帧图像， 建议160帧图像
        """
        self.cnt = 0
        self.nose_frames = []
        for src_frame in src_frames:
            nose_frame = self.get_nose_frame(src_frame)
            self.nose_frames.append(nose_frame)

    def calc_rr(self, res):  # 使用模型预测
        return get_rr_with_estimate(res, self.fps)

    def get_valid_segments(self, res):
        return get_valid_segments(res, self.fps)

    def get_predict_labels(self, src_frames):
        self.get_nose_frames(src_frames)
        labels = self._get_predict_labels()
        return labels

    def get_predict_labels_with_crop_frames(self, src_frames):
        self.nose_frames = src_frames
        labels = self._get_predict_labels()
        return labels

    def _get_predict_labels(self):
        num_frame = len(self.nose_frames)
        no_overlap_window_num = int(num_frame / self.chunk_len)
        overlap_window_num = 0 if no_overlap_window_num * self.chunk_len == num_frame else 1
        res = []
        self.nose_frames = resize_raw_crop_frames(self.nose_frames, self.image_size)
        with torch.no_grad():
            for i in range(no_overlap_window_num):
                chunk_frames = self.nose_frames[i * self.chunk_len: (i + 1) * self.chunk_len]  # [T, H, W, C]
                chunk_tensor = apply_transform_to_video(chunk_frames, self.transform).unsqueeze(0).to(
                    self.device)

                pred = self.model(chunk_tensor)  # [B, 2, T]
                pred.requires_grad_(False)
                _, predicted = torch.max(pred, 1)  # predicted [1, T]  每列要么是0要么是1
                res += list(predicted.cpu().flatten().numpy())

            if overlap_window_num == 1:
                chunk_frames = self.nose_frames[-self.chunk_len:]
                chunk_tensor = torch.from_numpy(apply_transform_to_video(chunk_frames, self.transform)).unsqueeze(0).to(
                    self.device)
                pred = self.model(chunk_tensor)  # [B, 2, T]
                pred.requires_grad_(False)
                _, predicted = torch.max(pred, 1)  # predicted [1, T]  每列要么是0要么是1
                res += list(predicted.cpu().flatten().numpy())[no_overlap_window_num * self.chunk_len - num_frame:]
        return res


if __name__ == '__main__':
    raw_data_path = r'thermalVid_2024_12_25_14_36_10_1'
    save_data_path = r'label.npy'
    calc_rr = RRDetectorModel(fps=25)