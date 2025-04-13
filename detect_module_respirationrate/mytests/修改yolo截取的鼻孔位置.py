#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/10/19 11:07
# @Author  : lqh
# @python-version 3.10
# @Software: PyCharm

    通过GUI界面调整YOLO模型截取的鼻子区域的大小
    通过该程序可以获得较好地截取鼻子区域的参数
"""
import os
import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QSpinBox, QLabel, QFileDialog, \
    QGridLayout, QFrame, QDoubleSpinBox
from PyQt5.QtGui import QPixmap, QImage, qRgb, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import ultralytics
import torch
from ultralytics import YOLO
import rr_detect_config

model_path = rr_detect_config.NOSEDETECTOR_MODEL_PATH
RESIZE_H = 96


class YOLOUtil:
    def __init__(self):
        self.model = YOLO(model_path, task="pose")  # pretrained YOLOv8n model
        self.testOne()  # 打通pipeline

    def testOne(self):
        self.model.predict(save=False, imgsz=640)  # return a list of Results objects

    def detect(self, np_img):
        results = self.model.predict(np_img, save=False, imgsz=640, verbose=False)  # return a list of Results objects
        if len(results) > 0:
            result = results[0]
            if len(result.boxes) != 0:
                # xmin, ymin, xmax, ymax = [item.int().item() for item in result.boxes.xyxy[0]]  # tensor转换为python类型的普通int
                # rect = ((xmin, ymin), (xmax, ymax))
                n_rect = np.array(result.boxes.xyxyn[0].cpu())  # (4,) 归一化的坐标
                n_face_w = result.boxes.xywhn[0][2].item()  # 归一化的脸宽
                n_face_h = result.boxes.xywhn[0][3].item()  # 归一化的脸高
                # cv2.rectangle(np_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                n_keypoints = np.array(result.keypoints.xyn[0].cpu())  # (5,2)
                # xys = keypoints.xy[0].int()
                # keypoints = []
                # for (x, y) in xys:
                #     keypoints.append((x.item(), y.item()))
                #     # cv2.circle(np_img, (x.item(), y.item()), 1, (0, 255, 0), 4)
                return n_rect, n_keypoints, n_face_w, n_face_h

        return (), [], 0, 0

    # 使用生成器函数，节省内存
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


yolo_util = YOLOUtil()


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QGridLayout()

        # 添加一个按钮来选择文件
        btn_open = QPushButton('选择 .npy 文件', self)
        btn_open.clicked.connect(self.openFile)
        layout.addWidget(btn_open, 0, 0)

        # 输入框用于显示文件路径
        self.file_line_edit = QLineEdit()
        layout.addWidget(self.file_line_edit, 0, 1)

        # 添加一个SpinBox控件
        self.index_spin_box = QSpinBox()
        self.index_spin_box.valueChanged.connect(self.update_image)
        self.index_spin_box.setValue(0)
        layout.addWidget(QLabel('索引:'), 1, 0)
        layout.addWidget(self.index_spin_box, 1, 1)

        self.label_spin_box = QSpinBox()
        self.label_spin_box.setRange(-1, 100)
        # self.label_spin_box.setValue(-1)
        self.label_label = QLabel('标签(无用)')
        layout.addWidget(self.label_label, 2, 0)
        layout.addWidget(self.label_spin_box, 2, 1)

        self.width_spin_box = QDoubleSpinBox()
        self.width_spin_box.setRange(0, 1)
        # 设置小数点后位数
        self.width_spin_box.setDecimals(2)  # 设置小数点后最多两位
        # 可选：设置单步增量
        self.width_spin_box.setSingleStep(0.01)  # 每次改变值的增量为0.01
        self.width_spin_box.setValue(0.14)
        layout.addWidget(QLabel('Width:'), 3, 0)
        layout.addWidget(self.width_spin_box, 3, 1)

        self.heightTop_spin_box = QDoubleSpinBox()
        self.heightTop_spin_box.setRange(0, 1)
        # 设置小数点后位数
        self.heightTop_spin_box.setDecimals(2)  # 设置小数点后最多两位
        # 可选：设置单步增量
        self.heightTop_spin_box.setSingleStep(0.01)  # 每次改变值的增量为0.01
        self.heightTop_spin_box.setValue(0.01)
        layout.addWidget(QLabel('Height_top:'), 4, 0)
        layout.addWidget(self.heightTop_spin_box, 4, 1)

        self.heightBottom_spin_box = QDoubleSpinBox()
        self.heightBottom_spin_box.setRange(0, 1)
        # 设置小数点后位数
        self.heightBottom_spin_box.setDecimals(2)  # 设置小数点后最多两位
        # 可选：设置单步增量
        self.heightBottom_spin_box.setSingleStep(0.01)  # 每次改变值的增量为0.01
        self.heightBottom_spin_box.setValue(0.16)
        layout.addWidget(QLabel('Height_bottom:'), 5, 0)
        layout.addWidget(self.heightBottom_spin_box, 5, 1)

        self.btn_yolo = QPushButton("yolo detect", self)
        self.btn_yolo.clicked.connect(self.yolo_detect)
        layout.addWidget(self.btn_yolo, 6, 0)

        # Matplotlib画布
        # self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # layout.addWidget(self.canvas, 3, 0, 1, 2)
        # self.axes = self.canvas.figure.subplots()

        # 设置主窗口布局
        self.setLayout(layout)
        self.setWindowTitle('Numpy Array Viewer')
        # 设置拖放功能
        self.setAcceptDrops(True)

        # 初始化数据
        self.data = None
        self.labels = None

        self.id_to_labelNames = {1: 'in', 2: 'out'}
        # 创建窗口
        self.cv2_window_name = 'thermal face'
        cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)

        # 添加键盘事件监听器
        self.keyPressEvent = self.onKeyPress

        self.show()

    def onKeyPress(self, event):
        if self.data is not None:
            index = self.index_spin_box.value()
            if event.key() == Qt.Key_A:
                new_index = max(index - 1, 0)
                self.index_spin_box.setValue(new_index)
            elif event.key() == Qt.Key_D:
                new_index = min(index + 1, self.data.shape[0] - 1)
                self.index_spin_box.setValue(new_index)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def update_data_and_label(self, data_file_path):
        self.file_line_edit.setText(data_file_path)
        self.data = np.load(data_file_path)
        self.data = self.data.astype(np.uint8)
        self.index_spin_box.setRange(0, self.data.shape[0] - 1)

        # # 获取文件所在的目录
        # directory = os.path.dirname(data_file_path)
        #
        # # 获取文件的基础名（不含路径），然后去除文件扩展名
        # basename = os.path.basename(data_file_path)
        # name_without_ext = os.path.splitext(basename)[0]
        # sign_name = name_without_ext.replace('vidNose', 'labelNose')
        # label_file_name = sign_name + '.txt'
        # label_file_path = os.path.join(directory, label_file_name)
        # if os.path.exists(label_file_path):
        #     self.labels = self.label_to_ndarray(label_file_path)
        cv2.moveWindow(self.cv2_window_name, 300, 300)
        self.index_spin_box.setValue(0)
        self.update_image(self.index_spin_box.value())

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = str(urls[0].toLocalFile())
            self.update_data_and_label(file_path)

    def label_to_ndarray(self, label_txt_path):
        with open(label_txt_path, 'r', encoding='utf-8') as file:
            txt_content = file.read()
            # 将字符串内容分割成行
            lines = txt_content.strip().split('\n')
            # 跳过第一行
            lines = lines[1:]
            # 确定数组的最大长度，其实这里可以直接定为500帧的
            last_line = lines[-1]
            max_frame = int(last_line.split(',')[1])

            # 创建一个全零的一维数组，长度为最大帧号加1（因为帧号从0开始）
            labels_array = np.zeros(max_frame + 1, dtype=np.uint8)

            # 填充数组
            for line in lines:
                start, end, label = map(int, line.split(','))
                labels_array[start:end + 1] = label
            return labels_array

    def openFile(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择 .npy 文件", "", "Numpy Files (*.npy);;All Files (*)",
                                                   options=options)
        if file_name:
            self.update_data_and_label(file_name)

    def yolo_detect(self):
        frame_index = self.index_spin_box.value()
        bgr_img = self.data[frame_index]
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        n_rect, n_kpt, n_face_w, n_face_h = yolo_util.detect(bgr_img)
        face_size = [0, 0]
        if len(n_rect) == 0:
            print('no face detect!')
            return
        rect = (
            (round(n_rect[0] * width), round(n_rect[1] * height)),
            (round(n_rect[2] * width), round(n_rect[3] * height))
        )
        kpt = []
        face_size[0] = n_face_w * width
        face_size[1] = n_face_h * height
        for p in n_kpt:
            kpt.append((round(p[0] * width), round(p[1] * height)))

        # 存入鼻子周围的热像图帧
        r_w, r_ht, r_hb = self.width_spin_box.value(), self.heightTop_spin_box.value(), self.heightBottom_spin_box.value()
        half_nose_width, nose_height_top, nose_height_bottom = face_size[0] * r_w, face_size[1] * r_ht, face_size[
            1] * r_hb
        nose_start_x = np.clip(round(kpt[2][0] - half_nose_width), 0, width)
        nose_start_y = np.clip(round(kpt[2][1] - nose_height_top), 0, height)
        nose_end_x = np.clip(round(kpt[2][0] + half_nose_width), 0, width)
        nose_end_y = np.clip(round(kpt[2][1] + nose_height_bottom), 0, height)
        bgr_img_copy = bgr_img.copy()
        cv2.rectangle(bgr_img_copy, (nose_start_x, nose_start_y), (nose_end_x, nose_end_y), (0, 0, 255), 2)
        cv2.imshow(self.cv2_window_name, bgr_img_copy)

    def update_image(self, index):
        image = self.data[index]
        # image = image.astype(np.uint8)
        # # 将浮点型数据转换为 0-255 范围内的 uint8 类型
        # if image.dtype == np.float32 or image.dtype == np.float64:
        #     # 检查数组中是否有大于 1 的元素
        #     if np.any(image > 1):
        #         image = image.astype(np.uint8)
        #     else:
        #         image = (image * 255).astype(np.uint8)
        # pil_image = Image.fromarray(image)
        # pixmap = self.convert_pil_to_qpixmap(pil_image)
        cv2.imshow(self.cv2_window_name, image)
        # self.axes.imshow(image)
        # self.canvas.draw_idle()
        # self.update_label(index)

    # def update_label(self, index):
    #     label_id = self.labels[index]
    #     self.label_spin_box.setValue(label_id)
    #     self.label_label.setText(self.id_to_labelNames[label_id])

    @staticmethod
    def convert_pil_to_qpixmap(pil_image):
        if pil_image.mode == "RGB":
            r, g, b = pil_image.split()
            pil_image = Image.merge("RGB", (b, g, r))
        elif pil_image.mode == "RGBA":
            r, g, b, a = pil_image.split()
            pil_image = Image.merge("RGBA", (b, g, r, a))
        elif pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        qimage = QImage(pil_image.tobytes(), pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
