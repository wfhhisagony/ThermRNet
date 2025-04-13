#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/10/26 22:13
# @Author  : lqh
# @python-version 3.10
# @Software: PyCharm
# @Description
    通过小球的放缩以恒定的呼吸速率(呼吸率)引导用户呼吸，球变大时用户吸气，球变小时用户呼气，控制球从最小变为最大、从最大变为最小的时间，从而使得用户的呼吸保持在恒定的速率
    可以通过此程序测试实时呼吸率检测的效果

"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QSpinBox, QLineEdit
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import Qt, QPropertyAnimation, QVariant, QEasingCurve, pyqtSlot
import random

DEFAULT_ANIMATION_TIME = 2500

class AnimatedBall(QWidget):
    def __init__(self, parent=None, animation_time=DEFAULT_ANIMATION_TIME):
        super().__init__(parent)
        self.resize(300, 300)
        self.ball_size = 50
        self.ball_color = Qt.red
        self.animation_time = animation_time
        self.start_animation()

    @pyqtSlot(QVariant)
    def on_value_changed(self, value):
        self.ball_size = value
        self.update()  # 强制重新绘制

    def start_animation(self):
        self.animation = QPropertyAnimation(self, b"ball_size")
        self.animation.valueChanged.connect(self.on_value_changed)

        self.animation.setEasingCurve(QEasingCurve.InOutQuad)  # 设置动画曲线
        self.animation.setDuration(self.animation_time)  # 动画持续时间为2秒
        self.animation.setStartValue(1)
        self.animation.setEndValue(300)
        self.animation.finished.connect(self.reverse_animation)
        self.animation.start()

    def reverse_animation(self):
        # self.animation.setDirection(QPropertyAnimation.Backward)
        startValue = self.animation.startValue()
        endValue = self.animation.endValue()
        self.animation.setStartValue(endValue)
        self.animation.setEndValue(startValue)
        # self.animation.setDuration(2000)  # 反向动画也持续2秒
        # self.animation.setStartValue(300)
        # self.animation.setEndValue(50)
        # self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        # self.animation.valueChanged.connect(self.on_value_changed)
        # self.animation.finished.connect(self.start_animation)
        self.animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.ball_color))
        center_x = int(self.rect().center().x())
        center_y = int(self.rect().center().y())
        radius = int(self.ball_size / 2)

        painter.drawEllipse(center_x - radius, center_y - radius, self.ball_size, self.ball_size)

    def set_ball_size(self, size):
        self.ball_size = size
        self.update()


class AnimatedVarBall(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(300, 300)
        self.ball_size = 50
        self.ball_color = Qt.red
        self.animation_times = [2500, 1500, 857]  # 相当于呼吸:12次/分钟,20次/分钟,35次/分钟
        self.cnt = 0
        self.start_animation()


    @pyqtSlot(QVariant)
    def on_value_changed(self, value):
        self.ball_size = value
        self.update()  # 强制重新绘制

    def start_animation(self):
        self.animation = QPropertyAnimation(self, b"ball_size")
        self.animation.valueChanged.connect(self.on_value_changed)

        self.animation.setEasingCurve(QEasingCurve.InOutQuad)  # 设置动画曲线
        self.animation.setDuration(random.choice(self.animation_times))  # 动画持续时间
        # self.cnt = (self.cnt + 1) % len(self.animation_times)
        self.animation.setStartValue(1)
        self.animation.setEndValue(300)
        self.animation.finished.connect(self.reverse_animation)
        self.animation.start()

    def reverse_animation(self):
        # self.animation.setDirection(QPropertyAnimation.Backward)
        startValue = self.animation.startValue()
        endValue = self.animation.endValue()
        self.animation.setStartValue(endValue)
        self.animation.setEndValue(startValue)
        self.animation.setDuration(random.choice(self.animation_times))  # 动画持续时间
        # self.cnt = (self.cnt + 1) % len(self.animation_times)
        # self.animation.setDuration(2000)  # 反向动画也持续2秒
        # self.animation.setStartValue(300)
        # self.animation.setEndValue(50)
        # self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        # self.animation.valueChanged.connect(self.on_value_changed)
        # self.animation.finished.connect(self.start_animation)
        self.animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.ball_color))
        center_x = int(self.rect().center().x())
        center_y = int(self.rect().center().y())
        radius = int(self.ball_size / 2)

        painter.drawEllipse(center_x - radius, center_y - radius, self.ball_size, self.ball_size)

    def set_ball_size(self, size):
        self.ball_size = size
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.default_animation_times = [2500, 1500, 857]
        self.setWindowTitle('主窗口')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()
        button = QPushButton('开始动画', self)
        button.clicked.connect(self.open_animated_ball_window)
        layout.addWidget(button)

        self.time_spinbox = QSpinBox(self)
        self.time_spinbox.setRange(0, 10000)
        self.time_spinbox.setSingleStep(100)
        self.time_spinbox.setValue(DEFAULT_ANIMATION_TIME)
        layout.addWidget(self.time_spinbox)

        lineEdit = QLineEdit(self)
        lineEdit.setText(' '.join(str(number) for number in self.default_animation_times))
        layout.addWidget(lineEdit)

        button = QPushButton('随机动画', self)
        button.clicked.connect(self.open_animated_var_ball_window)
        layout.addWidget(button)




        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_animated_ball_window(self):
        self.animated_ball_window = AnimatedBall(None, self.time_spinbox.value())
        self.animated_ball_window.show()

    def open_animated_var_ball_window(self):
        self.animated_ball_window = AnimatedVarBall()
        self.animated_ball_window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())