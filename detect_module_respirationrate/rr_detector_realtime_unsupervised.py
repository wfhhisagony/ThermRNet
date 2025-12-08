#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/6/15 15:28
# @Author  : lqh
# @python-version 3.10
# @File    : my_interface02.py
# @Software: PyCharm
# @Description:
    RRDetectorRealtime主进程控制呼吸率模型进程实现基于实时热像仪捕获的视频流进行呼吸率检测
此版本只适用于局域网网络良好时,PC端接收到安卓端热像仪的热像视频流帧率稳定时
"""
import copy
import os
import sys
import logging
import rr_detect_config
from signal_filters import post_process

# 创建日志记录器
logger = logging.getLogger()
# 设置日志记录器的级别
logger.setLevel(rr_detect_config.LOGGING_LEVEL)
logging.basicConfig(level=rr_detect_config.LOGGING_LEVEL, format="%(levelname)s::%(funcName)s:  %(message)s",
                    stream=sys.stdout)

from collections import deque
import numpy as np
import json
import cv2
import base64
import time
import socket
import threading

from rr_detector import MyYoloPoseModel

THERMAL_FRAME_WIDTH = 192  # 安卓端通过网络传入PC服务器的热成像图片分辨率
THERMAL_FRAME_HEIGHT = 256

THERMAL_FPS = 25
DETECT_FRAME_NUM_CIRCLE = THERMAL_FPS * 60  # 约1分钟  本研究中热像仪存在的缺陷，需要定时校准。每隔1分钟校准热像仪
DETECT_FRAME_NUM = THERMAL_FPS * 12  # 约12s的滑动窗口
STRIDE = DETECT_FRAME_NUM // 2  # 使得后续检测约6s进行一次

# 定义一个窗口
CV2_WINDOW_NAME = 'Cropped Nose Frame'
cv2.namedWindow(CV2_WINDOW_NAME, cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口


class RRDetectorRealtime:
    def __init__(self, thermal_frame_width=THERMAL_FRAME_WIDTH, thermal_frame_height=THERMAL_FRAME_HEIGHT,
                 thermal_fps=THERMAL_FPS, detect_frame_num=DETECT_FRAME_NUM, stride=STRIDE, rr_method=rr_detect_config.RRMethods.Peak,
                 signal_filter=rr_detect_config.FilterMethods.savgol):
        self.rect = (  (0, 0), (1, 1)  )
        self.kpt = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.face_size = [1, 1]
        self.thermal_frame_width = thermal_frame_width
        self.thermal_frame_height = thermal_frame_height

        self.thermal_fps = thermal_fps
        self.detect_frame_num = detect_frame_num
        self.stride = stride
        # self.face_detect_freq = self.stride
        self.face_detect_freq = THERMAL_FPS
        self.rr_method = rr_method
        self.signal_filter = signal_filter
        self.face_detect_model = MyYoloPoseModel()

        self.sk_writer_yolo, self.sk_reader_yolo = None, None
        self.conn_reader_yolo, self.conn_writer_yolo = None, None
        self.sk_writer_cmd, self.sk_reader_cmd = None, None
        self.conn_reader_cmd, self.conn_writer_cmd = None, None

        self.socket_reader_port_yolo = 30000  # 从PC端利用YOLO模型检测，手机端需要发送热像图给PC端，PC端再将热像图检测结果（特征点位置）发给手机端
        self.socket_writer_port_yolo = 30001
        self.socket_reader_port_cmd = 30002
        self.socket_writer_port_cmd = 30003

        self.stop_recv_flag = False
        self.nose_frames = []  # 多线程共享资源
        # self.nose_frames_lock = threading.Lock()  # python 多线程由于GIL锁，关于基本数据的基本操作都是原子操作，但要注意多个原子操作的组成的序列就不是原子操作了

        self.rr_deque = deque(maxlen=10)   # 可换成multiprocessing中的Queue，从而让其他进程接受


    def write_data(self, conn, data):
        """
        向conn(socket)发送数据data，data转换为字符串，先发送data字符串的长度，再发送data字符串
        @param:
            conn: socket object
            data: python dict
        """
        # 将dict转换为JSON字符串
        json_str = json.dumps(data)
        encoded_data = json_str.encode('utf-8')
        # 发送JSON字符串的长度（用于Java端确定读取多少字节）
        conn.sendall(len(encoded_data).to_bytes(4, byteorder='big'))  # 计算机默认使用小端，小端排序，字符串表现为左边低位，右边高位。这里应配合接收端的解析方法
        # 发送JSON字符串
        conn.sendall(encoded_data)

    def send_cmd(self, raw_cmd_string):
        """
            PC端发送用户定义好的命令字符串，由Android端解析字符串的信息，并执行相应操作
        """
        cmd_string = f"{raw_cmd_string}\n".encode('utf-8')
        self.conn_writer_cmd.sendall(cmd_string)

    def close_socket_connect_yolo(self):
        try:
            if self.sk_writer_yolo is not None:
                self.sk_writer_yolo.shutdown(socket.SHUT_RDWR)  # 通知对方不再发送数据, 禁止读RD和禁止写WR
                self.sk_writer_yolo.close()  # 关闭 socket
                self.conn_writer_yolo.close()
            if self.sk_reader_yolo is not None:
                self.sk_reader_yolo.shutdown(socket.SHUT_RDWR)
                self.sk_reader_yolo.close()
                self.conn_reader_yolo.close()
        except:
            pass
        self.sk_writer_yolo = None
        self.conn_writer_yolo = None
        self.sk_reader_yolo = None
        self.conn_reader_yolo = None

    def new_socket_connection_yolo(self):
        # global sk_writer_yolo, sk_reader_yolo, conn_writer_yolo, conn_reader_yolo
        # 创建服务器端套接字
        self.sk_writer_yolo = socket.socket()
        self.sk_reader_yolo = socket.socket()
        self.server_address = '0.0.0.0'  # 注意Client和Server对应通信Socket端口要一致
        # 把地址绑定到套接字

        self.sk_reader_yolo.bind((self.server_address, self.socket_reader_port_yolo))
        self.sk_writer_yolo.bind((self.server_address, self.socket_writer_port_yolo))
        # 监听连接
        self.sk_reader_yolo.listen()
        self.sk_writer_yolo.listen()
        print("YOLO 等待连接")
        # 接受客户端连接
        self.conn_reader_yolo, _ = self.sk_reader_yolo.accept()
        self.conn_writer_yolo, _ = self.sk_writer_yolo.accept()
        return self.conn_reader_yolo, self.conn_writer_yolo

    def close_socket_connect_cmd(self):
        # global sk_writer_cmd, sk_reader_cmd, conn_writer_cmd, conn_reader_cmd
        try:
            if self.sk_writer_cmd is not None:
                self.sk_writer_cmd.shutdown(socket.SHUT_RDWR)
                self.sk_writer_cmd.close()
                self.conn_writer_cmd.close()
            if self.sk_reader_cmd is not None:
                self.sk_reader_cmd.shutdown(socket.SHUT_RDWR)
                self.sk_reader_cmd.close()
                self.conn_reader_cmd.close()
        except:
            pass
        self.sk_writer_cmd = None
        self.conn_writer_cmd = None
        self.sk_reader_cmd = None
        self.conn_reader_cmd = None

    def new_socket_connection_cmd(self):
        # global sk_writer_cmd, sk_reader_cmd, conn_writer_cmd, conn_reader_cmd
        # 创建服务器端套接字
        self.sk_writer_cmd = socket.socket()
        self.sk_reader_cmd = socket.socket()
        server_address = '0.0.0.0'  # 注意Client和Server对应通信Socket端口要一致
        # 把地址绑定到套接字

        self.sk_reader_cmd.bind((server_address, self.socket_reader_port_cmd))
        self.sk_writer_cmd.bind((server_address, self.socket_writer_port_cmd))
        # 监听连接
        self.sk_reader_cmd.listen()
        self.sk_writer_cmd.listen()
        print("CMD 等待连接")
        # 接受客户端连接
        self.conn_reader_cmd, _ = self.sk_reader_cmd.accept()
        self.conn_writer_cmd, _ = self.sk_writer_cmd.accept()
        return self.conn_reader_cmd, self.conn_writer_cmd

    def recvall(self, sock, count):
        """
        接受count个字节
        @param:
            sock:  socket_reader
            count:  num of bytes data to receive
        @return:
            buf:  bytes stream
        """
        buf = b''  # buf是一个byte类型
        while count:
            # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def respond_client(self):
        width = self.thermal_frame_width  # Android Client端定义的发送的数据格式
        height = self.thermal_frame_height
        self.stop_recv_flag = False
        cnt = 0  # 每10次检测一次

        while not self.stop_recv_flag:
            start = time.time()  # 用于计算帧率信息
            length = self.recvall(self.conn_reader_yolo, 16)  # 获得图片文件的长度,16代表获取长度  byte的数目  默认使用UTF-8读取
            try:
                length = int(length)
                # print(length)
            except Exception as e:
                print(e)
                break
            stringData = self.recvall(self.conn_reader_yolo, int(length))  # 根据获得的文件长度，获取图片文件
            data_encode = base64.b64decode(stringData)  # 将获取到的字符流数据转换成1维数组
            jpegdata = np.frombuffer(data_encode, np.uint8)  # 一维数组 Jpeg格式，Jpeg数据中已包含图像的宽高等信息
            bgr_img = cv2.imdecode(jpegdata, cv2.IMREAD_COLOR)  # imdecode专门用来解析压缩成的jpeg数据
            if cnt % self.face_detect_freq == 0:
                n_rect, n_kpt, n_face_w, n_face_h = self.face_detect_model.detect(bgr_img)
                if len(n_rect) != 0:
                    self.rect = (
                        (round(n_rect[0] * width), round(n_rect[1] * height)),
                        (round(n_rect[2] * width), round(n_rect[3] * height))
                    )

                    self.face_size[0] = n_face_w * width
                    self.face_size[1] = n_face_h * height
                    for n_kpt_index, n_p in enumerate(n_kpt):
                        self.kpt[n_kpt_index][0], self.kpt[n_kpt_index][1] = round(n_p[0] * width), round(
                            n_p[1] * height)
                    n_rect_str = ",".join(map(str, n_rect.flatten().tolist()))
                    n_kpt_str = ",".join(map(str, n_kpt.flatten().tolist()))
                    dict_data = {
                        "rect": n_rect_str,
                        "kpt": n_kpt_str,
                        "w": n_face_w,
                        "h": n_face_h,
                    }
                    # threading.Thread(target=write_data, args=(self.conn_writer_yolo, dict_data,)).start()
                    self.write_data(self.conn_writer_yolo, dict_data)
            # if len(self.nose_frames) == 0:
            #     last_time = time.time()  # 时间戳(s)
            # self.time_list.append(time.time())

            # 存入鼻子周围的热像图帧
            half_nose_width, nose_height_top, nose_height_bottom = self.face_size[0] * 0.14, self.face_size[
                1] * 0.01, \
                                                                   self.face_size[1] * 0.16
            nose_start_x = np.clip(round(self.kpt[2][0] - half_nose_width), 0, width)
            nose_start_y = np.clip(round(self.kpt[2][1] - nose_height_top), 0, height)
            nose_end_x = np.clip(round(self.kpt[2][0] + half_nose_width), 0, width)
            nose_end_y = np.clip(round(self.kpt[2][1] + nose_height_bottom), 0, height)

            nose_frame = bgr_img[nose_start_y:nose_end_y, nose_start_x:nose_end_x]

            # 展示截取的鼻子区域
            cv2.imshow(CV2_WINDOW_NAME, nose_frame)
            cv2.waitKey(1)

            self.nose_frames.append(nose_frame)  # 这里也可以直接把nose_frame处理成raw_signal再加入

            if len(self.nose_frames) >= self.detect_frame_num:  # 检测一次
                threading.Thread(target=self.process_cropFrames, args=(copy.deepcopy(self.nose_frames), self.thermal_fps,)).start()

                self.nose_frames = self.nose_frames[self.stride:]  # 清空，防止一直递增占用内存
            # 绘制关键点
            # cv2.rectangle(bgr_img, self.rect[0], self.rect[1], (0, 0, 255), 2)
            # for point in self.kpt:
            #     cv2.circle(bgr_img, point, 1, (0, 255, 0), 4)
            # cv2.imshow("xx", bgr_img)
            # cv2.waitKey(1)
            # end = time.time()
            # seconds = end - start
            # try:
            #     fps = 'FPS:' + str(int(1 / seconds))  # 可能会导致seconds为0
            #     # print(fps)  # 这个操作会造成一定程度的卡顿
            # except Exception as e:
            #     print(e)
            cnt += 1

    def recv_record(self):
        length = self.recvall(self.conn_reader_cmd, 16)  # 获得图片文件的长度,16代表获取长度  byte的数目  默认使用UTF-8读取
        try:
            length = int(length)
            print(length)
        except:
            print("recv_record length读取失败")
        stringData = self.recvall(self.conn_reader_cmd, length)  # 根据获得的文件长度，获取图片文件
        data_bytes = base64.b64decode(stringData)  # 将获取到的字符流数据转换成1维数组
        record_data = str(data_bytes, 'utf-8')
        return record_data

    def process_cropFrames(self, crop_frames, fps):
        raw_signal = self.get_raw_signal_with_cropFrames(crop_frames, fps)
        smoothed_signal = self.get_smoothed_signal_with_rawSignal(raw_signal, fps, self.signal_filter)
        rr = self.get_rr(smoothed_signal, fps, self.rr_method)
        print(f'rr: {rr}')
        self.rr_deque.append(rr)

    def get_smoothed_signal_with_rawSignal(self, raw_signal, fps=25,
                                            signal_filter=rr_detect_config.FilterMethods.bandpass):
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

    def get_rr(self, smoothed_signal, fps, rr_method):
        if rr_method == rr_detect_config.RRMethods.FFT:
            rr = post_process.get_rr_with_fft(smoothed_signal, fps)
        elif rr_method == rr_detect_config.RRMethods.Peak:
            rr = post_process.get_rr_with_peak_distance(smoothed_signal, fps)
        else:
            raise Exception(f'No such rr_method: {rr_method}')
        return rr

    def get_raw_signal_with_cropFrames(self, crop_frames, fps=None):
        # TODO: 使用更好地提取raw呼吸信号的方法
        raw_signal = [np.mean(crop_frame) for crop_frame in crop_frames]  # 使用整张图片的像素平均值作为呼吸信号
        return raw_signal


if __name__ == '__main__':
    rr_method = rr_detect_config.RRMethods.Peak
    signal_filter = rr_detect_config.FilterMethods.savgol
    rr_detector_realtime = RRDetectorRealtime(rr_method=rr_method, signal_filter=signal_filter)

    while True:
        try:
            rr_detector_realtime.new_socket_connection_yolo()
            rr_detector_realtime.new_socket_connection_cmd()
            print("连接已建立")
            cmd_str = f"2:3,{DETECT_FRAME_NUM_CIRCLE}"
            rr_detector_realtime.send_cmd(cmd_str)
            rr_detector_realtime.respond_client()
        except ConnectionResetError:
            cv2.destroyAllWindows()
            rr_detector_realtime.close_socket_connect_yolo()
            rr_detector_realtime.close_socket_connect_cmd()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break
