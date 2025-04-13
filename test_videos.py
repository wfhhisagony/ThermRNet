import os.path

from physiology_detector import PhysiologyDetector

import cv2
import numpy as np
from collections import deque


def get_video_frames(video_path):
    # 创建一个 VideoCapture 对象
    cap = cv2.VideoCapture(video_path)
    # 检查是否成功打开视频文件
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")
    else:
        # 获取视频的帧率和总帧数
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 读取第一帧以获取帧的高度和宽度
        ret, frame = cap.read()
        if not ret:
            raise Exception("Error: Could not read the first frame.")
        else:
            height, width, channels = frame.shape

            # 初始化一个空的 NumPy 数组来存储所有帧
            frames_array = np.empty((total_frames, height, width, channels), dtype=np.uint8)

            # 将第一帧添加到数组中
            frames_array[0] = frame

            # 读取剩余的帧并添加到数组中
            for i in range(1, total_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"Error: Could not read frame {i}.")
                    break
                frames_array[i] = frame
        return frames_array, fps


def rr_test():
    print('rr_test')
    physiology_detector = PhysiologyDetector()
    model_list = ['rr']
    physiology_detector.init(model_list)

    # warning: Donnot use too long video, suggest you to read crop frames not raw video file
    vid_dir = r"X:\DNPMM_v1.1"
    # warning: Donnot use too long video, suggest you to read crop frames not raw video file
    for i in range(3):
        vid_path = os.path.join(vid_dir, f'output_thermal_{i + 1}.mp4')
        print(vid_path)
        src_frames, fps = get_video_frames(vid_path)
        window_size = 160
        for k in range(len(src_frames) // window_size):
            rr = physiology_detector.get_rr(src_frames[k * window_size:(k + 1) * window_size], fps)
            print(rr)


if __name__ == '__main__':
    rr_test()
