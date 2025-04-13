from detect_module_respirationrate.rr_detector import NoseCropper
from physiology_detector import PhysiologyDetector

import threading
import cv2
import time
import numpy as np
from queue import Queue
from scipy import signal

flag = 0
# 原始数据点个数
N = 160

# 缓存数据个数
N1 = 1
num_fft = N
data_queue = Queue(N)
HBS_queue = Queue(N1)
thermal_resize_size = 96

nose_cropper = NoseCropper()
thermal_data_list = []


def rr_test():
    print('rr_test')
    physiology_detector = PhysiologyDetector()
    model_list = ['rr']
    physiology_detector.init(model_list)

    vid_path = "rtsp://xxxxx@192.168.1.64:554/Streaming/Channels/201"
    cap = cv2.VideoCapture(vid_path)
    fps = 25
    # 定义一个窗口以便于观察
    CV2_WINDOW_NAME = 'Cropped Nose Frame'
    cv2.namedWindow(CV2_WINDOW_NAME, cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口

    if not cap.isOpened():
        raise ValueError("无法打开摄像头")
    # 等待摄像头预热
    time.sleep(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 截取鼻子
        crop_frame = nose_cropper.get_crop_frame(frame)

        crop_frame = cv2.resize(crop_frame, (thermal_resize_size, thermal_resize_size),
                                interpolation=cv2.INTER_LINEAR)  # 差不多用INTER_AREA

        cv2.rectangle(frame, (nose_cropper.last_col, nose_cropper.last_row), (
            nose_cropper.last_col + nose_cropper.last_width, nose_cropper.last_row + nose_cropper.last_height),
                      color=(255, 0, 0), thickness=1)
        cv2.imshow('ori', frame)
        cv2.imshow(CV2_WINDOW_NAME, crop_frame)
        cv2.waitKey(1)
        thermal_data_list.append(crop_frame)
        if len(thermal_data_list) == N:
            rr = physiology_detector.get_rr_with_crop_frames(thermal_data_list, fps)
            print(rr)
            thermal_data_list.clear()


if __name__ == '__main__':
    rr_test()
