from detect_module_respirationrate import RRDetectorModel, rr_detect_config

import cv2
import numpy as np
from collections import deque


class PhysiologyDetector:
    def __init__(self, max_queue_len=5, rr_queue_len=480):
        # 滤波队列的大小
        self.max_queue_len = max_queue_len
        self.kf_flag = False

        self.rr_queue_len = rr_queue_len

    def init(self, model_list=['hr', 'rr', 'bt'], **kwargs):

        if 'rr' in model_list:
            # 根据实际填写thermal_camera的参数
            self.thermal_img_width = 192 if kwargs.get('thermal_img_width') is None else kwargs['thermal_img_width']
            self.thermal_img_height = 256 if kwargs.get('thermal_img_height') is None else kwargs['thermal_img_height']
            self.thermal_fps = 25 if kwargs.get('thermal_fps') is None else kwargs['thermal_fps']
            self.rr_detector = RRDetectorModel(width=self.thermal_img_width, height=self.thermal_img_height,
                                               fps=self.thermal_fps)
            self.rr_labels = deque(maxlen=self.rr_queue_len)

    def get_rr(self, src_frames, fps):
        """
        如果使用这个方法必须保证RRDetector设置正确的相机帧FPS
        @param
            src_frames:  original thermal camera frames 原始图像的数组，不需要经过鼻部截取 160帧图片以上，建议160帧图像
            fps:  sample rate of the thermal camera
        @return
            rr: 呼吸率
        """
        labels = self.rr_detector.get_predict_labels(src_frames)
        self.rr_labels.extend(labels)
        # 引入权重，后面检测出来的更重要
        valid_segments = self.rr_detector.get_valid_segments(list(self.rr_labels))
        valid_segments_len = len(valid_segments)
        if valid_segments_len == 0:
            rr = 0
        else:
            if valid_segments_len % 2 != 0:
                if valid_segments_len == 1:  # 当只存在一个呼或吸的帧数时，用相同的数目补充一个吸或呼的帧数，使其为一个完整的呼吸
                    valid_segments = np.append(valid_segments, valid_segments[-1])
                    valid_segments_len += 1
                else:  # 当帧数的数目为奇数时，说明多出了一次呼或吸的帧数，这时舍去第一个，只取完整的呼吸
                    valid_segments = valid_segments[1:]
            if valid_segments_len > 3:
                valid_segments_len = valid_segments_len // 2
                rr = fps * 30 / (
                    (np.mean(valid_segments[:valid_segments_len]) * 0.25 +
                     np.mean(valid_segments[valid_segments_len:]) * 0.75)
                )
            else:
                rr = fps * 30 / np.mean(valid_segments)
        return rr

    def get_rr_with_crop_frames(self, src_frames, fps):
        """
       @param
           src_frames:  crop nose frames 建议160帧图像
           fps:  sample rate of the thermal camera
       @return
           rr: 呼吸率
       """
        labels = self.rr_detector.get_predict_labels_with_crop_frames(src_frames)
        self.rr_labels.extend(labels)
        # 引入权重，后面检测出来的更重要
        valid_segments = self.rr_detector.get_valid_segments(list(self.rr_labels))
        valid_segments_len = len(valid_segments)
        if valid_segments_len == 0:
            rr = 0
        else:
            if valid_segments_len % 2 != 0:
                if valid_segments_len == 1:  # 当只存在一个呼或吸的帧数时，用相同的数目补充一个吸或呼的帧数，使其为一个完整的呼吸
                    valid_segments = np.append(valid_segments, valid_segments[-1])
                    valid_segments_len += 1
                else:  # 当帧数的数目为奇数时，说明多出了一次呼或吸的帧数，这时舍去第一个，只取完整的呼吸
                    valid_segments = valid_segments[1:]
            if valid_segments_len > 3:
                valid_segments_len = valid_segments_len // 2
                rr = fps * 30 / (
                    (np.mean(valid_segments[:valid_segments_len]) * 0.25 + np.mean(
                        valid_segments[valid_segments_len:]) * 0.75))
            else:
                rr = fps * 30 / np.mean(valid_segments)
        return rr


if __name__ == '__main__':
    def rr_test():
        print('rr_test')
        physiology_detector = PhysiologyDetector()
        model_list = ['rr']
        physiology_detector.init(model_list)

        # warning: Donnot use too long video, suggest you to read crop frames not raw video file
        vid_path = r"./detect_module_respirationrate/mytests/test_thermalVideo.mp4"

        def get_video_frames(video_path):
            fps = 25
            frames_array = np.load(video_path)
            return frames_array, fps

        src_frames, fps = get_video_frames(vid_path)
        window_size = 160
        for i in range(len(src_frames) // window_size):
            rr = physiology_detector.get_rr(src_frames[i * window_size:(i + 1) * window_size], fps)
            print(rr)


    rr_test()
