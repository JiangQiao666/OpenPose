import csv
import math
import re
import sys

# 添加在import区
import random
import traceback

import matplotlib.pyplot as plt
import requests
import dashscope
from http import HTTPStatus
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import sqlite3

from PyQt5.QtGui import *
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QMessageBox, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QDateEdit, QCheckBox, \
    QSizePolicy, QStackedLayout
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, pyqtSignal, QDate, Qt, QSize, QUrl, QRectF, QPoint, QPropertyAnimation, QThread
import cv2
import numpy as np
import time
import threading
from queue import Queue
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QFrame
# # 导入 UserProfileWindow 类
# from .user_profile import UserProfileWindow
# OpenPose初始化
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    if sys.platform == "win32":
        sys.path.append(dir_path + '/../../python/openpose/Release')
        os.environ['PATH'] += ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        sys.path.append('../../python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found.')
    raise e

# 更新动作任务配置
TASK_CONFIG = {
    "Bobath握手练习": {
        "target_angle_range": (80, 120),
        "voice_prompts": {
            "too_low": "请再抬高一点手臂，加油！",
            "good_posture": "动作非常标准，继续保持！",
            "overbend": "请注意不要过度弯曲哦!",
            "low_confidence": "请将双手保持在画面中",
            "rep_complete": "已经成功完成{}次标准动作！继续加油！"
        },
        "joint_ids": {"shoulder": 2, "elbow": 3, "wrist": 4},
        "video_path": r"D:\文档\我的视频\Bobath握手训练.mp4"  # 添加视频路径
    },
    "桥式运动": {
        "target_angle_range": (30, 60),
        "voice_prompts": {
            "too_low": "请再抬高一点臀部，保持动作！",
            "good_posture": "动作标准，保持良好！",
            "overbend": "注意不要过度后仰！",
            "low_confidence": "请确保臀部在画面中",
            "rep_complete": "完成得非常好！已经成功完成{}次！"
        },
        "joint_ids": {"hip": 9, "knee": 10},
        "video_path": r"D:\文档\我的视频\bridge.mp4"
    },
    "关节活动度训练": {
        "target_angle_range": (120, 160),
        "voice_prompts": {
            "too_low": "请再活动关节，保持动作！",
            "good_posture": "关节活动良好，继续保持！",
            "overbend": "注意不要过度活动关节！",
            "low_confidence": "请确保关节在画面中",
            "rep_complete": "完成得非常好！已经成功完成{}次！"
        },
        "joint_ids": {"hip": 9, "knee": 10, "ankle": 11},
        "video_path": r"D:\文档\我的视频\joint.mp4"
    },
    "坐位神经滑动练习": {
        "target_angle_range": (45, 90),
        "voice_prompts": {
            "too_low": "请再向前弯腰，保持动作！",
            "good_posture": "神经滑动练习标准，保持良好！",
            "overbend": "注意不要过度弯腰！",
            "low_confidence": "请确保腰部在画面中",
            "rep_complete": "完成得非常好！已经成功完成{}次！"
        },
        "joint_ids": {"spine": 1, "hip": 9},
        "video_path": r"D:\文档\我的视频\nerve_slide.mp4"
    },
    "伸展运动": {
        "target_angle_range": (60, 120),
        "voice_prompts": {
            "too_low": "请再伸展，保持动作！",
            "good_posture": "伸展动作标准，保持良好！",
            "overbend": "注意不要过度伸展！",
            "low_confidence": "请确保身体在画面中",
            "rep_complete": "完成得非常好！已经成功完成{}次！"
        },
        "joint_ids": {"shoulder": 2, "hip": 9},
        "video_path": r"D:\文档\我的视频\stretch.mp4"
    },
}

# 康复参数配置
REHAB_SETTINGS = {
    "confidence_threshold": 0.3,
    "ui_color_correct": (46, 204, 113),
    "ui_color_wrong": (231, 76, 60),
    "rep_duration": 4,
    "voice_interval": 1,
    "rep_voice_interval": 1,
    "volume": 0.9
}

#个人信息对话框类
class UserProfileWindow(QDialog):
    def __init__(self, user_id, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.setWindowTitle("个人信息")
        self.setGeometry(400, 300, 400, 500)

        # 设置整体样式
        self.setStyleSheet("""
            /* 主窗口背景 - 现代磨砂玻璃效果 */
            QDialog {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(245, 247, 250, 0.98),
                    stop:1 rgba(228, 232, 240, 0.95)
                );
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
            }

            /* 标题区域 - 沉浸式设计 */
            QLabel#headerTitle {
                font: 24px 'Segoe UI Semibold';
                color: #2D3748;
                letter-spacing: 1px;
                padding-bottom: 8px;
                border-bottom: 2px solid rgba(66, 153, 225, 0.15);
            }

            /* 信息分组 - 新拟态设计 */
            QGroupBox {
                background: rgba(255, 255, 255, 0.85);
                border: none;
                border-radius: 14px;
                margin: 16px 0;
                padding: 20px;
                box-shadow: 
                    8px 8px 16px rgba(0, 0, 0, 0.04),
                    -8px -8px 16px rgba(255, 255, 255, 0.6);
            }

            /* 输入控件 - 现代极简风 */
            QLineEdit, QComboBox {
                border: 1px solid rgba(203, 213, 225, 0.5);
                border-radius: 8px;
                padding: 12px;
                font: 15px 'Segoe UI';
                color: #4A5568;
                background: rgba(255, 255, 255, 0.9);
                min-height: 40px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            QLineEdit:focus, QComboBox:focus {
                border-color: #4299E1;
                box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.2);
            }

            /* 按钮系统 - 现代渐变方案 */
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4299E1, 
                    stop:1 #3182CE
                );
                color: white;
                border: none;
                border-radius: 8px;
                padding: 14px 24px;
                font: bold 15px 'Segoe UI';
                text-transform: uppercase;
                letter-spacing: 0.5px;
                transition: all 0.2s ease;
            }

            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4299E1, 
                    stop:1 #2B6CB0
                );
                transform: translateY(-1px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            /* 数据标签 - 强调色系统 */
            QLabel[data-type="label"] {
                color: #718096;
                font: 14px 'Segoe UI Semibold';
            }

            QLabel[data-type="value"] {
                color: #2D3748;
                font: 15px 'Segoe UI';
                padding: 4px 0;
            }

            /* 底部信息栏 - 玻璃质感 */
            QLabel#footer {
                background: rgba(255, 255, 255, 0.85);
                border-top: 1px solid rgba(226, 232, 240, 0.6);
                color: #718096;
                padding: 12px;
                font: 13px 'Segoe UI';
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # 添加顶部标题区域
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(15)

        # 假设你的自定义图标图片路径是 "path/to/your/avatar.png"
        avatar_path = "D:/文档/我的图片/个人信息.png"

        # 创建 QLabel 用于显示头像
        avatar_label = QLabel()
        avatar_label.setFixedSize(72, 72)  # 设置固定大小

        # 设置头像图片
        avatar_pixmap = QPixmap(avatar_path)  # 加载图片
        avatar_pixmap = avatar_pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 按比例缩放
        avatar_label.setPixmap(avatar_pixmap)  # 设置图片到 QLabel

        # # 修改头像标签样式（可选，根据需要调整）
        # avatar_label.setStyleSheet("""
        # QLabel {
        #     border-radius: 36px;
        #     background: qradialgradient(
        #         cx:0.5, cy:0.5, radius: 0.8,
        #         fx:0.3, fy:0.3,
        #         stop:0 #FFFFFF, stop:1 #3498db
        #     );
        #     border: 2px solid #ffffff;
        #     box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        # }
        # """)

        # 设置居中对齐
        avatar_label.setAlignment(Qt.AlignCenter)

        # 标题
        title_label = QLabel("个人信息")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        title_label.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(avatar_label)
        header_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(header_frame)

        # 个人信息表单
        self.info_group = QGroupBox("个人信息")
        info_layout = QFormLayout()
        info_layout.setSpacing(10)
        info_layout.setLabelAlignment(Qt.AlignRight)

        # 创建标签
        def create_label(text, data_type="value"):
            label = QLabel()
            label.setProperty("data-type", data_type)
            return label

        self.username_label = create_label("用户名", "value")
        self.age_label = create_label("年龄", "value")
        self.age_group_label = create_label("年龄段", "value")
        self.gender_label = create_label("性别", "value")
        self.height_label = create_label("身高", "value")
        self.weight_label = create_label("体重", "value")
        self.bmi_label = create_label("BMI", "value")


        # 创建标签
        username_lbl = QLabel("用户名:")
        username_lbl.setProperty("data-type", "label")
        age_lbl = QLabel("年龄:")
        age_lbl.setProperty("data-type", "label")
        age_group_lbl = QLabel("年龄段:")
        age_group_lbl.setProperty("data-type", "label")
        gender_lbl = QLabel("性别:")
        gender_lbl.setProperty("data-type", "label")
        height_lbl = QLabel("身高 (cm):")
        height_lbl.setProperty("data-type", "label")
        weight_lbl = QLabel("体重 (kg):")
        weight_lbl.setProperty("data-type", "label")
        bmi_lbl = QLabel("BMI:")
        bmi_lbl.setProperty("data-type", "label")

        info_layout.addRow(username_lbl, self.username_label)
        info_layout.addRow(age_lbl, self.age_label)
        info_layout.addRow(age_group_lbl, self.age_group_label)
        info_layout.addRow(gender_lbl, self.gender_label)
        info_layout.addRow(height_lbl, self.height_label)
        info_layout.addRow(weight_lbl, self.weight_label)
        info_layout.addRow(bmi_lbl, self.bmi_label)

        self.info_group.setLayout(info_layout)
        main_layout.addWidget(self.info_group)

        # 添加修改按钮
        self.modify_button = QPushButton("修改信息", self)
        self.modify_button.clicked.connect(self.show_modify_dialog)
        main_layout.addWidget(self.modify_button)

        self.modify_button.setStyleSheet("""
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 15px;
            font-weight: bold;
            transition: background-color 0.3s, transform 0.2s;
        }

        QPushButton:hover {
            background-color: #2980b9;
            transform: scale(1.02);
        }
        """)

        # 添加底部信息
        footer_label = QLabel("© 2025 用户信息 - 所有权利保留")
        footer_label.setStyleSheet("font-size: 14px; color: #7f8c8d;")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)

        footer_label.setStyleSheet("""
        font-size: 14px;
        color: #7f8c8d;
        margin-top: 20px;
        """)

        self.load_user_data()

    #更改个人信息窗口方法
    def show_modify_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("修改个人信息")
        dialog.setModal(True)
        dialog.setMinimumSize(400, 600)  # 设置最小窗口尺寸

        # 主样式表（带渐变背景和柔和阴影）
        dialog.setStyleSheet("""
        QDialog {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                      stop:0 #f5f7fa, stop:1 #c3cfe2);
            border-radius: 12px;
            padding: 24px;
            font-family: 'Microsoft YaHei';
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }

        QLabel {
            color: #4a4a4a;
            font-size: 14px;
            min-width: 80px;
        }

        QLineEdit, QComboBox {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #dcdcdc;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 14px;
            selection-background-color: #a8d8ff;
            min-height: 28px;
        }

        QLineEdit:focus, QComboBox:hover {
            border-color: #7ec2ff;
            background: white;
            box-shadow: 0 2px 6px rgba(126, 194, 255, 0.2);
        }

        QPushButton {
            background-color: #7ec2ff;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 24px;
            font-size: 14px;
            transition: all 0.3s;
        }

        QPushButton:hover {
            background-color: #6ab0f5;
            transform: translateY(-1px);
        }

        QPushButton:pressed {
            background-color: #5a9de3;
            transform: translateY(0);
        }

        QPushButton#cancel_btn {
            background-color: #ff7e7e;
        }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)

        # 表单布局
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(15)
        form_layout.setHorizontalSpacing(20)
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.old_password = QLineEdit()  # 添加这行
        self.new_password = QLineEdit()  # 添加这行
        self.confirm_password = QLineEdit()  # 添加这行

        # 创建带占位符的输入框
        self.new_username = QLineEdit()
        self.new_username.setPlaceholderText("2-16位字符")
        self.new_age = QLineEdit()
        self.new_age.setValidator(QIntValidator(1, 150))  # 年龄验证
        self.new_gender = QComboBox()
        self.new_gender.addItems(["男", "女", "其他"])
        self.new_height = QLineEdit()
        self.new_weight = QLineEdit()

        # 初始化密码相关控件
        self.old_password = QLineEdit()  # 添加这行
        self.new_password = QLineEdit()  # 添加这行
        self.confirm_password = QLineEdit()  # 添加这行
        self.new_gender.setCursor(Qt.PointingHandCursor)
        # 其他控件初始化...

        # 密码输入框单独样式
        for pwd_edit in [self.old_password, self.new_password, self.confirm_password]:
            pwd_edit.setPlaceholderText("至少6位字符")
            pwd_edit.setStyleSheet("letter-spacing: 2px;")

        # 添加到表单布局
        form_layout.addRow(QLabel("👤 新用户名:"), self.new_username)
        form_layout.addRow(QLabel("🎂 新年龄:"), self.new_age)
        form_layout.addRow(QLabel("🚻 新性别:"), self.new_gender)
        form_layout.addRow(QLabel("📏 新身高:"), self.new_height)
        form_layout.addRow(QLabel("⚖️ 新体重:"), self.new_weight)
        form_layout.addRow(QLabel("🔑 旧密码:"), self.old_password)  # 添加这行
        form_layout.addRow(QLabel("🔒 新密码:"), self.new_password)  # 添加这行
        form_layout.addRow(QLabel("🔒 确认密码:"), self.confirm_password)  # 添加这行

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        save_button = QPushButton("💾 保存修改")
        save_button.setCursor(Qt.PointingHandCursor)
        cancel_button = QPushButton("❌ 取消", objectName="cancel_btn")
        cancel_button.setCursor(Qt.PointingHandCursor)

        # 按钮点击效果
        save_button.clicked.connect(lambda: dialog.accept())
        cancel_button.clicked.connect(lambda: dialog.reject())

        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)

        # 组装界面
        layout.addLayout(form_layout)
        layout.addStretch()
        layout.addLayout(button_layout)

        # 显示动画
        fade_in = QPropertyAnimation(dialog, b"windowOpacity")
        fade_in.setDuration(200)
        fade_in.setStartValue(0)
        fade_in.setEndValue(1)
        fade_in.start()

        if dialog.exec_() == QDialog.Accepted:
            self.update_user_data()

    def update_user_data(self):
        new_username = self.new_username.text()
        new_age = self.new_age.text()
        new_gender = self.new_gender.currentText()
        new_height = self.new_height.text()
        new_weight = self.new_weight.text()
        old_password = self.old_password.text()
        new_password = self.new_password.text()
        confirm_password = self.confirm_password.text()

        # 在修改对话框的输入框中添加验证器
        self.new_age.setValidator(QIntValidator(1, 150, self))
        self.new_height.setValidator(QDoubleValidator(0, 300, 2, self))
        self.new_weight.setValidator(QDoubleValidator(0, 600, 2, self))

        # 实时验证提示
        self.new_password.textChanged.connect(lambda: self.validate_password_strength())
        self.confirm_password.textChanged.connect(lambda: self.check_password_match())

        # 验证新密码
        if new_password != confirm_password:
            QMessageBox.warning(self, "警告", "新密码和确认密码不一致！")
            return

        # 这里添加数据库更新逻辑
        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()

        # 更新用户基本信息
        if new_username or old_password:
            if old_password:
                # 验证旧密码是否正确（假设数据库中有password字段）
                cursor.execute("SELECT password FROM Users WHERE user_id = ?", (self.user_id,))
                stored_password = cursor.fetchone()
                if stored_password and stored_password[0] != old_password:
                    QMessageBox.warning(self, "警告", "旧密码不正确！")
                    conn.close()
                    return

            cursor.execute(
                "UPDATE Users SET username = ?, password = ? WHERE user_id = ?",
                (new_username or None, new_password or None, self.user_id)
            )

        # 更新用户详细信息
        if new_age or new_gender or new_height or new_weight:
            cursor.execute(
                """
                UPDATE PatientProfiles
                SET age = ?, gender = ?, height = ?, weight = ?
                WHERE user_id = ?
                """,
                (new_age or None, new_gender or None, new_height or None, new_weight or None, self.user_id)
            )

        conn.commit()
        conn.close()

        # 更新界面显示
        self.load_user_data()
        QMessageBox.information(self, "成功", "个人信息已更新！")

    def calculate_password_strength(password):
        if not password:
            return 0.0

        strength = 0.0
        length = len(password)

        # --------------------------
        # 基础评分（最高50分）
        # --------------------------
        # 长度评分（0-30分）
        length_score = min(30, length * 2)  # 每字符得2分，上限30分（15字符）

        # 字符多样性评分（0-20分）
        type_bonus = 0
        types = {
            'lower': re.search(r'[a-z]', password),
            'upper': re.search(r'[A-Z]', password),
            'digit': re.search(r'\d', password),
            'special': re.search(r'[^A-Za-z0-9]', password)
        }
        type_count = sum(1 for t in types.values() if t)
        type_bonus = 5 * type_count  # 每类字符得5分，最高20分

        base_score = length_score + type_bonus

        # --------------------------
        # 扣分项（最多-20分）
        # --------------------------
        # 重复字符模式（如"aaa"）
        repeat_deduction = 0
        if re.search(r'(.)\1{2,}', password):  # 3个及以上重复字符
            repeat_deduction -= 5

        # 连续序列（如"123"或"abc"）
        sequence_deduction = 0
        for i in range(len(password) - 2):
            a, b, c = map(ord, password[i:i + 3])
            if (a + 1 == b and b + 1 == c) or (a - 1 == b and b - 1 == c):
                sequence_deduction -= 5
                break

        # 仅单一字符类型
        if type_count == 1:
            repeat_deduction -= 5

        # 常见弱密码（示例列表可扩展）
        weak_passwords = [
            'password', '123456', 'qwerty',
            'admin', 'welcome', 'abc123'
        ]
        if password.lower() in weak_passwords:
            return 0.0

        # --------------------------
        # 最终强度计算（0.0 ~ 1.0）
        # --------------------------
        total_score = max(0, base_score + repeat_deduction + sequence_deduction)
        normalized = min(1.0, total_score / 50)  # 标准化到0.0-1.0

        # 非线性增强：强度曲线更陡峭
        strength = pow(normalized, 0.7)

        return round(strength, 2)

    def validate_password_strength(self):
        # 密码强度实时指示
        strength = self.calculate_password_strength(self.new_password.text())
        self.password_strength_indicator.update_strength(strength)

    def load_user_data(self):
        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()

        # 查询用户基本信息
        cursor.execute("SELECT username FROM Users WHERE user_id = ?", (self.user_id,))
        user = cursor.fetchone()
        if user:
            self.username_label.setText(user[0])

        # 查询用户详细信息
        cursor.execute("""
            SELECT age, gender, height, weight
            FROM PatientProfiles
            WHERE user_id = ?
        """, (self.user_id,))
        profile = cursor.fetchone()
        if profile:
            self.age_label.setText(str(profile[0]))
            self.gender_label.setText(profile[1])
            self.height_label.setText(str(profile[2]))
            self.weight_label.setText(str(profile[3]))
            # 计算BMI
            height = profile[2]
            weight = profile[3]
            if height and weight:
                height_m = height / 100  # 转换为米
                bmi = weight / (height_m ** 2)
                self.bmi_label.setText(f"{bmi:.2f}")

            # 判断并设置年龄段
            self.age_group_label.setText(self.determine_age_group(profile[0]))


        conn.close()

    def determine_age_group(self, age):
        try:
            if 0 <= age <= 2:
                return "婴儿 (0-2岁)"
            elif 3 <= age <= 6:
                return "幼儿 (3-6岁)"
            elif 7 <= age <= 12:
                return "儿童 (7-12岁)"
            elif 13 <= age <= 18:
                return "青少年 (13-18岁)"
            elif 19 <= age <= 35:
                return "青年 (19-35岁)"
            elif 36 <= age <= 60:
                return "中年 (36-60岁)"
            elif 61 <= age <= 70:
                return "低龄老年 (61-70岁)"
            elif 71 <= age <= 80:
                return "中龄老年 (71-80岁)"
            else:
                return "高龄老年 (81岁及以上)"
        except (ValueError, TypeError):
            return "未知"

#语音助手
class VoiceAssistant:
    def __init__(self):
        self.engine = None
        self.queue = Queue()
        self.last_play_time = 0
        self.last_rep_time = 0
        self._init_engine()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _init_engine(self):
        try:
            if sys.platform == "win32":
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('volume', REHAB_SETTINGS["volume"])
                self.engine.setProperty('rate', 220)
            print("[语音系统] 语音引擎初始化完成")
        except Exception as e:
            print(f"[语音系统] 初始化失败: {str(e)}")

    def speak(self, text):
        if time.time() - self.last_play_time > REHAB_SETTINGS["voice_interval"]:
            self.queue.put(text)
            self.last_play_time = time.time()

    def speak_rep_complete(self, text):
        if time.time() - self.last_rep_time > REHAB_SETTINGS["rep_voice_interval"]:
            self.queue.put(text)
            self.last_rep_time = time.time()
            self.last_play_time = time.time()

    def _run(self):
        while True:
            text = self.queue.get()
            try:
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    if sys.platform == "darwin":
                        os.system(f'say "{text}"')
                    elif sys.platform == "linux":
                        os.system(f'espeak -v zh "{text}" 2>/dev/null')
            except Exception as e:
                print(f"[语音系统] 播放失败: {str(e)}")


# 康复计数器类
class RehabilitationCounter:
    def __init__(self):
        # 初始化总完成次数为 0
        self.total_reps = 0
        # 初始化动作开始时间为 None
        self.rep_start_time = None
        # 初始化动作状态为 False，表示动作未进行
        self.is_rep_ongoing = False
        # 创建一个线程锁，用于确保线程安全
        self.lock = threading.Lock()

    def update(self, current_status):
        # 使用锁确保线程安全
        with self.lock:
            # 如果当前状态为 True（动作正在进行）
            if current_status:
                # 如果动作尚未开始
                if not self.is_rep_ongoing:
                    # 记录动作开始时间
                    self.rep_start_time = time.time()
                    # 标记为动作正在进行
                    self.is_rep_ongoing = True
            else:
                # 如果当前状态为 False（动作未进行），则停止动作
                self.is_rep_ongoing = False
            # 如果动作正在进行，并且持续时间达到设定的重复时间
            if self.is_rep_ongoing and (time.time() - self.rep_start_time) >= REHAB_SETTINGS["rep_duration"]:
                # 增加完成次数
                self.total_reps += 1
                # 标记为动作未进行
                self.is_rep_ongoing = False
                # 返回 True 表示完成了一次动作
                return True
            # 返回 False 表示未完成动作
            return False

    @property
    def count(self):
        # 使用锁确保线程安全
        with self.lock:
            # 返回总完成次数
            return self.total_reps

    @property
    def is_counting(self):
        # 使用锁确保线程安全
        with self.lock:
            # 返回是否正在执行动作
            return self.is_rep_ongoing


def calculate_joint_angle(a, b, c):
    # 将输入点转换为 NumPy 数组
    a, b, c = map(np.array, [a, b, c])
    # 计算向量之间的弧度差
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # 将弧度转换为角度
    angle = np.abs(radians * 180.0 / np.pi)
    # 返回最小的角度值
    return min(angle, 360 - angle)


def calculate_neck_angle(neck, shoulder):
    # 将输入点转换为 NumPy 数组
    neck = np.array(neck[:2])
    shoulder = np.array(shoulder[:2])
    # 定义垂直方向的向量
    vertical = np.array([0, 1])
    # 计算颈部和肩部之间的向量
    vector = neck - shoulder
    # 计算向量与垂直方向的弧度差
    radians = np.arctan2(vector[1], vector[0])
    # 将弧度转换为角度
    angle = np.abs(radians * 180.0 / np.pi)
    # 返回最小的角度值
    return min(angle, 360 - angle)


def calculate_leg_angle(hip, knee, ankle):
    # 将输入点转换为 NumPy 数组
    hip = np.array(hip[:2])
    knee = np.array(knee[:2])
    ankle = np.array(ankle[:2])
    # 计算向量之间的弧度差
    radians = np.arctan2(ankle[1] - knee[1], ankle[0] - knee[0]) - np.arctan2(hip[1] - knee[1], hip[0] - knee[0])
    # 将弧度转换为角度
    angle = np.abs(radians * 180.0 / np.pi)
    # 返回最小的角度值
    return min(angle, 360 - angle)

#患者详细信息对话框类
class UserDetailsDialog(QDialog):
    def __init__(self, user_id):
        super().__init__()
        self.setWindowTitle("患者详细信息")
        self.setFixedSize(800, 970)
        self.user_id = user_id

        # 高级医疗风格样式表
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                          stop:0 #F8FBFF, stop:1 #E6F3FF);
                font-family: 'Microsoft YaHei', 'Segoe UI';
            }
            QGroupBox {
                border: 2px solid rgba(52, 152, 219, 0.2);
                border-radius: 12px;
                margin-top: 20px;
                background: rgba(255, 255, 255, 0.95);
                padding: 15px 25px;
                box-shadow: 0 6px 12px rgba(52, 152, 219, 0.08);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                color: #2C3E50;
                font-size: 18px;
                font-weight: 400;
                padding: 2 2px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 rgba(52, 152, 219, 0.1), stop:1 rgba(52, 152, 219, 0.05));
                border-radius: 8px;
            }
            QLabel[data-type="label"] {
                font-size: 15px;
                font: bold;
                color: #7F8C8D;
                min-width: 120px;
                padding: 2px 0;
            }
            QLabel[data-type="value"] {
                font-size: 15px;
                color: #2C3E50;
                padding: 2px 2px;
                border-radius: 2px;
                background: rgba(236, 240, 243, 0.6);
                border: 1px solid rgba(52, 152, 219, 0.15);
                min-width: 200px;
            }
            QLabel[data-type="alert"] {
                color: #E74C3C;
                font-weight: 500;
            }
            #healthIndicator {
                qproperty-alignment: AlignCenter;
                font-size: 14px;
                padding: 6px 12px;
                border-radius: 15px;
                min-width: 200px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(35, 35, 35, 35)
        main_layout.setSpacing(25)

        # 顶部医疗风格标题栏
        header = QFrame()
        header.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                             "stop:0 #3498DB, stop:1 #2C77CE); border-radius: 12px;")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(25, 25, 25, 25)

        title = QLabel("患者详细信息")
        title.setStyleSheet("""
            font-size: 24px;
            color: white;
            font-weight: bold;
            qproperty-alignment: AlignCenter;
        """)

        # header_layout.addWidget(medical_icon, 0, Qt.AlignCenter)
        header_layout.addWidget(title)
        main_layout.addWidget(header)

        # 数据展示区
        content_layout = QVBoxLayout()
        content_layout.setSpacing(18)

        # 动态生成信息组
        groups = [
            ("基本信息", [
                ("用户名:", "username_label"),
                ("年龄:", "age_label"),
                ("年龄段:", "age_group_label"),
                ("性别:", "gender_label"),
                ("身高 (cm):", "height_label"),
                ("体重 (kg):", "weight_label"),
                ("BMI:", "bmi_label")
            ]),
            ("联系信息", [
                ("家庭地址:", "address_label"),
                ("电话:", "phone_label"),
                ("紧急联系人:", "emergency_contact_label"),
                ("紧急联系人电话:", "emergency_phone_label")
            ]),
            ("健康信息", [
                ("静息心率:", "resting_hr_label"),
                ("血压 (mmHg):", "blood_pressure_label"),
                ("血氧饱和度 (%):", "blood_oxygen_label"),
                ("病种分类:", "condition_label")
            ])
        ]

        for group_title, items in groups:
            group = QGroupBox(group_title)
            layout = QFormLayout()
            layout.setHorizontalSpacing(30)
            layout.setVerticalSpacing(12)

            for label_text, field_name in items:
                lbl = QLabel(label_text)
                lbl.setProperty("data-type", "label")

                value = QLabel()
                value.setProperty("data-type", "value")
                setattr(self, field_name, value)

                # 特殊字段处理
                if "血压" in label_text:
                    self.blood_pressure_indicator = QLabel()
                    self.blood_pressure_indicator.setObjectName("healthIndicator")
                    hbox = QHBoxLayout()
                    hbox.addWidget(value)
                    hbox.addWidget(self.blood_pressure_indicator)
                    layout.addRow(lbl, hbox)
                else:
                    layout.addRow(lbl, value)

            group.setLayout(layout)
            content_layout.addWidget(group)

        main_layout.addLayout(content_layout)

        # 健康状态指示器
        self.health_status = QLabel()
        self.health_status.setObjectName("healthIndicator")
        main_layout.addWidget(self.health_status, 0, Qt.AlignCenter)

        # 加载数据
        self.load_user_data()

    def load_user_data(self):
        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()

        # 查询用户基本信息
        cursor.execute("SELECT username FROM Users WHERE user_id = ?", (self.user_id,))
        user = cursor.fetchone()
        if user:
            self.username_label.setText(user[0])

        # 查询用户详细信息
        cursor.execute("""
            SELECT age, gender, height, weight, address, phone, emergency_contact, emergency_phone, 
                   resting_hr, blood_pressure, blood_oxygen, condition
            FROM PatientProfiles
            WHERE user_id = ?
        """, (self.user_id,))
        profile = cursor.fetchone()
        if profile:
            self.age_label.setText(str(profile[0]))
            self.gender_label.setText(profile[1])
            self.height_label.setText(str(profile[2]))
            self.weight_label.setText(str(profile[3]))
            self.address_label.setText(profile[4])
            self.phone_label.setText(profile[5])
            self.emergency_contact_label.setText(profile[6])
            self.emergency_phone_label.setText(profile[7])
            self.resting_hr_label.setText(str(profile[8]))
            self.blood_pressure_label.setText(profile[9])
            self.blood_oxygen_label.setText(str(profile[10]))
            self.condition_label.setText(profile[11])

            # 判断并设置年龄段
            self.age_group_label.setText(self.determine_age_group(profile[0]))

            # 计算BMI
            height = profile[2]
            weight = profile[3]
            if height and weight:
                height_m = height / 100  # 转换为米
                bmi = weight / (height_m ** 2)
                self.bmi_label.setText(f"{bmi:.2f}")

        # 新增健康状态判断
        self.update_health_indicators()

        conn.close()

    def update_health_indicators(self):
        """动态更新健康指标可视化"""
        try:
            # 血压可视化
            systolic = int(self.blood_pressure_label.text().split('/')[0])
            bp_color = "#2ECC71"  # 正常
            if systolic > 140:
                bp_color = "#E74C3C"  # 高血压
                self.blood_pressure_indicator.setStyleSheet(f"background: {bp_color}; color: white;")
                self.blood_pressure_indicator.setText("血压偏高")
            else:
                bp_color = "#2ECC71"  # 高血压
                self.blood_pressure_indicator.setStyleSheet(f"background: {bp_color}; color: white;")
                self.blood_pressure_indicator.setText("血压正常")
            # BMI可视化
            bmi = float(self.bmi_label.text())
            bmi_status = "正常" if 18.5 <= bmi <= 24 else ("过轻" if bmi < 18.5 else "超重")
            status_color = {
                "正常": "#2ECC71",
                "过轻": "#3498DB",
                "超重": "#E67E22"
            }.get(bmi_status, "#95A5A6")
            self.bmi_label.setStyleSheet(f"border-color: {status_color};")

            # 整体健康状态判断（新增多级判断）
            if bp_color == "#2ECC71":  # 血压正常时
                if bmi_status == "正常":
                    health_status = "健康状态：良好"
                    health_color = "#2ECC71"
                elif bmi_status == "过轻":
                    health_status = "注意：体重过轻\n建议营养咨询"
                    health_color = "#3498DB"
                else:  # 超重
                    health_status = "注意：体重超重\n建议运动管理"
                    health_color = "#E67E22"
            else:  # 血压异常时
                if bmi_status == "正常":
                    health_status = "建议：血压异常!!!"
                    health_color = "#E74C3C"
                else:
                    health_status = "警告：综合异常\n建议全面检查"
                    health_color = "#E74C3C"

            # 设置健康状态显示样式
            self.health_status.setStyleSheet(f"""
                background: {health_color};
                color: white;
                font-weight: 500;
                border-radius: 8px;
                padding: 8px 15px;
                min-width: 700px; 
            """)
            self.health_status.setText(health_status)
            self.health_status.setToolTip("点击查看详细建议")  # 添加悬浮提示

        except Exception as e:
            print(f"健康指标更新错误: {str(e)}")

    def determine_age_group(self, age):
        try:
            if 0 <= age <= 2:
                return "婴儿 (0-2岁)"
            elif 3 <= age <= 6:
                return "幼儿 (3-6岁)"
            elif 7 <= age <= 12:
                return "儿童 (7-12岁)"
            elif 13 <= age <= 18:
                return "青少年 (13-18岁)"
            elif 19 <= age <= 35:
                return "青年 (19-35岁)"
            elif 36 <= age <= 60:
                return "中年 (36-60岁)"
            elif 61 <= age <= 70:
                return "低龄老年 (61-70岁)"
            elif 71 <= age <= 80:
                return "中龄老年 (71-80岁)"
            else:
                return "高龄老年 (81岁及以上)"
        except (ValueError, TypeError):
            return "未知"

# 康复训练师窗口类
class TherapistWindow(QMainWindow):
    task_updated = pyqtSignal()  # 定义任务更新信号

    def __init__(self, login_window):
        super().__init__()
        self.setWindowTitle("康复训练系统 - 康复训练任务管理")
        self.setGeometry(450, 100, 1000, 800)
        self.login_window = login_window
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                font-family: 'Microsoft YaHei', sans-serif;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QComboBox, QLineEdit, QDateEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                min-height: 32px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #ccc;
            }
            QComboBox::down-arrow {
                image: url(:/icons/down_arrow.svg);
            }
            QPushButton {
                border: none;
                border-radius: 4px;
                padding: 10px 15px;
                font-weight: 500;
                cursor: pointer;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                gridline-color: #ddd;
            }
            QHeaderView::section {
                background-color: #e9ecef;
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QTableWidget::item:selected {
                background-color: #cce5ff;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #f5f9ff, stop:1 #e3f2fd);
            }
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                min-height: 100px;
            }
        """)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 标题栏
        header_layout = QHBoxLayout()
        header_label = QLabel("康复训练任务管理")
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        header_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(header_label)
        main_layout.addLayout(header_layout)

        # 创建任务发布区域 - 使用分组框
        task_group = QGroupBox("任务发布")
        task_layout = QVBoxLayout()
        task_layout.setSpacing(10)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # 用户选择
        user_layout = QHBoxLayout()
        user_layout.setSpacing(10)

        self.user_icon = QLabel()
        self.user_icon.setPixmap(QPixmap("D:\文档\我的图片\我的患者.png").scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # 添加“详细信息”点击事件
        self.user_icon.mousePressEvent = self.show_user_details
        self.user_label = QLabel("选择用户:")
        self.user_label.setStyleSheet(" font-size: 14px;padding: 4px 0; qproperty-alignment: AlignVCenter;")
        self.user_combo = QComboBox()
        self.load_users()
        self.user_combo.setMinimumWidth(180)
        self.user_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        user_layout.addWidget(self.user_icon)
        user_layout.addWidget(self.user_combo)
        form_layout.addRow(self.user_label, user_layout)

        # 任务描述
        # 创建一个水平布局，用于放置任务描述相关的控件
        task_layout_desc = QHBoxLayout()
        # 设置水平布局中控件之间的间距为10
        task_layout_desc.setSpacing(10)
        # 创建一个用于显示任务图标（图片）的标签
        task_icon = QLabel()
        # 设置任务图标为指定路径的图片，并将其缩放到宽高均为30像素，同时保持图片的宽高比例并进行平滑变换
        task_icon.setPixmap(QPixmap("D:\文档\我的图片\任务描述.png").scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # 创建一个用于显示任务描述文本的标签
        self.task_desc_label = QLabel("任务描述:")
        # 设置任务描述标签的样式，包括字体大小为14像素、上下内边距为4像素、文字垂直居中对齐
        self.task_desc_label.setStyleSheet(" font-size: 14px;padding: 4px 0; qproperty-alignment: AlignVCenter;")
        # 创建一个下拉框，用于选择任务描述
        self.task_desc_edit = QComboBox()
        # 将TASK_CONFIG字典的键添加到下拉框中，作为可选择的任务描述选项
        self.task_desc_edit.addItems(TASK_CONFIG.keys())
        # 设置下拉框的最小宽度为20像素
        self.task_desc_edit.setMinimumWidth(20)
        # 设置下拉框的大小策略为水平方向可扩展、垂直方向优先
        self.task_desc_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # # 将任务图标和任务描述下拉框添加到水平布局中
        task_layout_desc.addWidget(task_icon)
        task_layout_desc.addWidget(self.task_desc_edit)
        # 将任务描述标签和水平布局（包含任务图标和下拉框）作为一行添加到表单布局中
        form_layout.addRow(self.task_desc_label, task_layout_desc)

        # 目标重复次数

        reps_layout = QHBoxLayout()
        reps_layout.setSpacing(10)

        reps_icon = QLabel()
        reps_icon.setPixmap(QPixmap("D:\文档\我的图片\重复次数.png").scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        reps_icon.setFixedSize(32, 32)
        reps_icon.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.target_reps_label = QLabel("目标次数:")
        self.target_reps_label.setStyleSheet(" font-size: 14px;padding: 4px 0; qproperty-alignment: AlignVCenter;")
        self.target_reps_edit = QLineEdit()
        self.target_reps_edit.setPlaceholderText("请输入目标次数")
        self.target_reps_edit.setValidator(QIntValidator(1, 100))
        # self.target_reps_edit.setMaximumWidth(200)  # 限制输入框宽度
        # self.target_reps_edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        reps_layout.addWidget(reps_icon)
        reps_layout.addWidget(self.target_reps_edit)
        form_layout.addRow(self.target_reps_label, reps_layout)

        # 日期
        date_layout = QHBoxLayout()
        date_layout.setSpacing(10)

        date_icon = QLabel()
        date_icon.setPixmap(QPixmap("D:\文档\我的图片\截止时间.png").scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        date_icon.setFixedSize(32, 32)
        date_icon.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.due_date_label = QLabel("截止日期:")
        self.due_date_label.setStyleSheet(" font-size: 14px;padding: 4px 0; qproperty-alignment: AlignVCenter;")
        self.due_date_edit = QDateEdit()
        self.due_date_edit.setCalendarPopup(True)
        self.due_date_edit.setDate(QDate.currentDate().addDays(7))  # 默认设置为一周后
        self.due_date_edit.setMinimumWidth(150)
        self.due_date_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        date_layout.addWidget(date_icon)
        date_layout.addWidget(self.due_date_edit)
        form_layout.addRow(self.due_date_label, date_layout)

        # 添加表单布局到任务布局
        task_layout.addLayout(form_layout)

        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        self.publish_task_button = QPushButton("发布任务")
        self.publish_task_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.publish_task_button.clicked.connect(self.publish_task)

        button_layout.addWidget(self.publish_task_button)

        task_layout.addLayout(button_layout)
        task_group.setLayout(task_layout)
        main_layout.addWidget(task_group)

        # 用户训练情况表格区域
        table_group = QGroupBox("用户训练情况")
        table_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid rgba(52, 152, 219, 0.2);
                border-radius: 12px;
                margin-top: 15px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #f5f9ff, stop:1 #e3f2fd);
                padding: 15px;
            }
            QGroupBox::title {
                color: #2C3E50;
                font-size: 18px;
                padding: 2 2px;
                border-radius: 8px;
            }
        """)

        table_layout = QVBoxLayout()

        # 表格样式配置
        self.training_table = QTableWidget(0, 8)
        self.training_table.setObjectName("MedicalTable")
        self.training_table.setStyleSheet("""
            QTableWidget#MedicalTable {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                padding: 5px;
                gridline-color: rgba(52, 152, 219, 0.1);
                alternate-background-color: #f8f9ff;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #f5f9ff, stop:1 #e3f2fd);
                color: #2c3e50;
                border: none;
                padding: 12px;
                font-weight: 500;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid rgba(52, 152, 219, 0.1);
            }
            QTableWidget::item:selected {
                background: rgba(52, 152, 219, 0.15);
                color: #2c3e50;
            }
        """)

        # 表头设置
        self.training_table.setHorizontalHeaderLabels(["任务ID", "用户名", "任务描述", "目标次数", "完成次数", "日期", "状态", "User_id"])
        self.training_table.setColumnHidden(0, True)
        self.training_table.setColumnHidden(7, True)
        header = self.training_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setMinimumSectionSize(100)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # 表格行为设置
        self.training_table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格为只读，防止用户编辑内容
        self.training_table.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置选择行为为选择整行，提升用户体验
        self.training_table.setSelectionMode(QAbstractItemView.SingleSelection)  # 设置选择模式为单选，防止多行被同时选中
        self.training_table.verticalHeader().setVisible(False)  # 隐藏垂直表头（行号），使界面更简洁
        self.training_table.setAlternatingRowColors(True)  # 启用交替行颜色，增强阅读体验，使相邻行更容易区分
        self.training_table.setShowGrid(False)  # 隐藏网格线，使表格看起来更简洁

        # 设置行高和字体
        self.training_table.verticalHeader().setDefaultSectionSize(40)
        self.training_table.setFont(QFont("Microsoft YaHei", 10))

        # 状态列渲染代理（示例）
        class StatusDelegate(QStyledItemDelegate):
            def paint(self, painter, option, index):
                if index.column() == 6:
                    text = index.data(Qt.DisplayRole)
                    # 根据状态设置颜色
                    status_colors = {
                        "Completed": ("#2ecc71", "#ffffff"),
                        "In Progress": ("#3498db", "#ffffff"),
                        "Not Started": ("#95a5a6", "#ffffff")
                    }
                    bg_color, text_color = status_colors.get(text, ("#e74c3c", "#ffffff"))

                    painter.save()
                    painter.setRenderHint(QPainter.Antialiasing)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(bg_color))
                    painter.drawRoundedRect(option.rect.adjusted(2, 2, -2, -2), 5, 5)

                    painter.setPen(QColor(text_color))
                    painter.drawText(option.rect, Qt.AlignCenter, text)
                    painter.restore()
                else:
                    super().paint(painter, option, index)

        self.training_table.setItemDelegate(StatusDelegate())

        table_layout.addWidget(self.training_table)
        table_group.setLayout(table_layout)
        main_layout.addWidget(table_group)

        # 操作按钮区域
        operation_group = QGroupBox()
        operation_layout = QHBoxLayout()

        self.delete_task_button = QPushButton("删除任务")
        self.delete_task_button.setStyleSheet("background-color: #F44336; color: white;")
        self.delete_task_button.clicked.connect(self.delete_task)

        self.edit_task_button = QPushButton("修改任务")
        self.edit_task_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.edit_task_button.clicked.connect(self.edit_task)

        operation_layout.addWidget(self.delete_task_button)
        operation_layout.addWidget(self.edit_task_button)
        operation_group.setLayout(operation_layout)
        main_layout.addWidget(operation_group)

        # 底部区域
        footer_layout = QHBoxLayout()
        footer_layout.addStretch(1)

        self.back_button = QPushButton("返回登录")
        self.back_button.setStyleSheet("background-color: #6c757d; color: white;")
        self.back_button.clicked.connect(self.back_to_login)

        operation_layout.addWidget(self.back_button)
        # main_layout.addLayout(footer_layout)

        # 定时器，用于定期刷新任务数据
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_training_data)
        self.refresh_timer.start(5000)  # 每5秒刷新一次

        # 初始化数据
        self.load_training_data()

    def show_user_details(self,event):
        print("详细信息被点击")
        user_id = self.user_combo.currentData()
        if not user_id:
            QMessageBox.warning(self, "查看详细信息失败", "请选择一个用户！", QMessageBox.Ok)
            return

        dialog = UserDetailsDialog(user_id)
        dialog.exec_()

    def update_task_details(self):
        # 获取任务描述下拉框中当前选中的任务名称
        task_name = self.task_desc_edit.currentText()
        # 判断任务名称是否存在于TASK_CONFIG字典中
        if task_name in TASK_CONFIG:
            # 如果存在，获取该任务的详细信息
            task_info = TASK_CONFIG[task_name]
            # 从任务详细信息中提取目标角度范围
            angle_range = task_info["target_angle_range"]
            # 从任务详细信息中提取语音提示内容
            voice_prompts = task_info["voice_prompts"]

            # 初始化一个字符串变量，用于存储要显示的任务详细信息
            details = f"<b>任务名称:</b> {task_name}<br>"  # 添加任务名称
            details += f"<b>目标角度范围:</b> {angle_range[0]}° - {angle_range[1]}°<br><br>"  # 添加目标角度范围
            details += "<b>语音提示:</b><br>"  # 添加语音提示的标题
            # 遍历语音提示内容，将其添加到details字符串中
            for key, prompt in voice_prompts.items():
                details += f"  - {key}: {prompt}<br>"

            # 将整理好的任务详细信息设置到任务详细信息文本框中进行显示
            self.task_details_text.setText(details)

    def load_users(self):
        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username FROM Users WHERE role = '患者'")
        users = cursor.fetchall()
        conn.close()

        self.user_combo.clear()
        for user in users:
            self.user_combo.addItem(user[1], user[0])  # user[1]是用户名，user[0]是用户ID

    def publish_task(self):
        user_id = self.user_combo.currentData()
        if not user_id:
            QMessageBox.warning(self, "发布任务失败", "请选择一个用户！", QMessageBox.Ok)
            return

        task_desc = self.task_desc_edit.currentText()
        target_reps = self.target_reps_edit.text()
        due_date = self.due_date_edit.date().toString("yyyy-MM-dd")

        if not target_reps:
            QMessageBox.warning(self, "发布任务失败", "请输入目标重复次数！", QMessageBox.Ok)
            return

        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO RehabTasks (therapist_id, patient_id, task_description, target_reps, due_date, status) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (1, user_id, task_desc, target_reps, due_date, 'Not Started'))
            conn.commit()
            QMessageBox.information(self, "发布任务成功", "任务已成功发布！", QMessageBox.Ok)
            self.load_training_data()
            self.task_updated.emit()
        except Exception as e:
            QMessageBox.warning(self, "发布任务失败", f"发布任务时发生错误: {str(e)}", QMessageBox.Ok)
        finally:
            conn.close()

    def delete_task(self):
        selected_row = self.training_table.currentRow()
        if selected_row == -1:
            QMessageBox.warning(self, "删除任务失败", "请选择一个任务！", QMessageBox.Ok)
            return

        task_id = self.training_table.item(selected_row, 0).text()

        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM RehabTasks WHERE task_id = ?", (task_id,))
            conn.commit()
            if cursor.rowcount > 0:
                QMessageBox.information(self, "删除任务成功", "任务已成功删除！", QMessageBox.Ok)
                self.load_training_data()
                self.task_updated.emit()
            else:
                QMessageBox.warning(self, "删除任务失败", "未找到指定的任务ID！", QMessageBox.Ok)
        except Exception as e:
            QMessageBox.warning(self, "删除任务失败", f"删除任务时发生错误: {str(e)}", QMessageBox.Ok)
        finally:
            conn.close()

    def edit_task(self):
        selected_row = self.training_table.currentRow()
        if selected_row == -1:
            QMessageBox.warning(self, "修改任务失败", "请选择一个任务！", QMessageBox.Ok)
            return

        task_id = self.training_table.item(selected_row, 0).text()
        task_desc = self.training_table.item(selected_row, 2).text()
        target_reps = self.training_table.item(selected_row, 3).text()
        due_date = self.training_table.item(selected_row, 5).text()

        edit_dialog = QDialog(self)
        edit_dialog.setWindowTitle("修改任务")
        edit_dialog.setFixedSize(400, 350)
        edit_dialog.setStyleSheet(self.styleSheet())

        main_layout = QVBoxLayout(edit_dialog)

        # 任务描述
        task_desc_layout = QHBoxLayout()
        task_desc_icon = QLabel()
        task_desc_icon.setPixmap(
            QPixmap("D:\文档\我的图片\任务描述.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        task_desc_icon.setFixedSize(32, 32)
        task_desc_icon.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        task_desc_label = QLabel("任务描述:")
        self.task_desc_edit_dialog = QComboBox()
        self.task_desc_edit_dialog.addItems(TASK_CONFIG.keys())
        self.task_desc_edit_dialog.setCurrentText(task_desc)

        task_desc_layout.addWidget(task_desc_icon)
        task_desc_layout.addWidget(self.task_desc_edit_dialog)
        main_layout.addLayout(task_desc_layout)

        # 目标重复次数
        reps_layout_dialog = QHBoxLayout()
        reps_icon_dialog = QLabel()
        reps_icon_dialog.setPixmap(
            QPixmap("D:\文档\我的图片\重复次数.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        reps_icon_dialog.setFixedSize(32, 32)
        reps_icon_dialog.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        target_reps_label_dialog = QLabel("目标重复次数:")
        self.target_reps_edit_dialog = QLineEdit(target_reps)
        self.target_reps_edit_dialog.setValidator(QIntValidator(1, 100))

        reps_layout_dialog.addWidget(reps_icon_dialog)
        reps_layout_dialog.addWidget(self.target_reps_edit_dialog)
        main_layout.addLayout(reps_layout_dialog)

        # 截止日期
        date_layout_dialog = QHBoxLayout()
        date_icon_dialog = QLabel()
        date_icon_dialog.setPixmap(
            QPixmap("D:\文档\我的图片\截止时间.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        date_icon_dialog.setFixedSize(32, 32)
        date_icon_dialog.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        due_date_label_dialog = QLabel("截止日期:")
        self.due_date_edit_dialog = QDateEdit()
        self.due_date_edit_dialog.setCalendarPopup(True)
        self.due_date_edit_dialog.setDate(QDate.fromString(due_date, "yyyy-MM-dd"))

        date_layout_dialog.addWidget(date_icon_dialog)
        date_layout_dialog.addWidget(self.due_date_edit_dialog)
        main_layout.addLayout(date_layout_dialog)

        buttons_layout = QHBoxLayout()
        save_button = QPushButton("保存")
        save_button.setStyleSheet("background-color: #4CAF50; color: white;")
        save_button.clicked.connect(lambda: self.save_edit_task(edit_dialog, task_id))
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(edit_dialog.reject)
        buttons_layout.addWidget(save_button)
        buttons_layout.addWidget(cancel_button)

        main_layout.addLayout(buttons_layout)

        edit_dialog.exec_()

    def save_edit_task(self, dialog, task_id):
        task_desc = self.task_desc_edit_dialog.currentText()
        target_reps = self.target_reps_edit_dialog.text()
        due_date = self.due_date_edit_dialog.date().toString("yyyy-MM-dd")

        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE RehabTasks 
                SET task_description = ?, target_reps = ?, due_date = ?
                WHERE task_id = ?
            """, (task_desc, target_reps, due_date, task_id))
            conn.commit()
            QMessageBox.information(self, "修改任务成功", "任务已成功修改！", QMessageBox.Ok)
            dialog.accept()
            self.load_training_data()
            self.task_updated.emit()
        except Exception as e:
            QMessageBox.warning(self, "修改任务失败", f"修改任务时发生错误: {str(e)}", QMessageBox.Ok)
        finally:
            conn.close()
    #加载训练数据
    def load_training_data(self):
        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.task_id, u.username, t.task_description, t.target_reps, t.completed_reps, t.due_date, t.status,u.user_id 
            FROM RehabTasks t
            JOIN Users u ON t.patient_id = u.user_id
        """)
        training_data = cursor.fetchall()
        conn.close()

        # 断开之前的点击事件连接
        try:
            self.training_table.cellClicked.disconnect()
        except TypeError:
            pass  # 如果没有连接，则忽略错误

        self.training_table.setRowCount(len(training_data))
        for row, data in enumerate(training_data):
            task_id = data[0]
            username = data[1]
            task_desc = data[2]
            target_reps = data[3]
            completed_reps = data[4]
            print("加载训练任务")
            print(training_data)
            print("加载训练任务结束")
            due_date = data[5]
            status = data[6]
            user_id_for_clicked = data[7]

            self.training_table.setItem(row, 0, QTableWidgetItem(str(task_id)))
            self.training_table.setItem(row, 1, QTableWidgetItem(username))
            self.training_table.setItem(row, 2, QTableWidgetItem(task_desc))
            self.training_table.setItem(row, 3, QTableWidgetItem(str(target_reps)))
            self.training_table.setItem(row, 4, QTableWidgetItem(str(completed_reps)))
            self.training_table.setItem(row, 5, QTableWidgetItem(due_date))
            self.training_table.setItem(row, 6, QTableWidgetItem(status))
            self.training_table.setItem(row, 7, QTableWidgetItem(str(user_id_for_clicked)))
            # 设置点击事件来显示训练历史
        self.training_table.cellClicked.connect(self.show_user_training_history)
    #调用用户任务的训练历史
    def show_user_training_history(self, row, column):
        """
        显示用户训练历史的函数，当用户点击任务状态列时触发。
        参数:
            row (int): 被点击的行号。
            column (int): 被点击的列号。
        """
        # 检查是否点击了状态列（第7列，索引从0开始）
        if column != 6:
            return  # 如果不是状态列，则不执行后续操作

        # 获取当前行中状态列的单元格内容
        status_item = self.training_table.item(row, 6)
        # print("当前行的状态")
        # print(status_item.text())
        # print("当前用状态打印结束")
        # 检查任务状态是否为“进行中”或“已完成”，如果不是则不显示历史记录
        if status_item.text() not in ["In Progress", "Completed"]:
            return

        # 获取当前行的用户名（第2列）
        user_name = self.training_table.item(row, 1).text()
        user_id_for_clicked2 = self.training_table.item(row, 7).text()
        print("当前行的用户名")
        print(user_name)
        print("当前用户名结束")
        # user_name = self.training_table.item(row, 7).text()
        print("当前行的用户id")
        print(user_id_for_clicked2)
        print("当前用户名id结束")

        # 获取当前行的任务描述（第3列）
        # task_description = self.training_table.item(row, 2).text()
        #
        # # 获取当前行的用户ID（假设用户ID存储在第1列，根据实际结构调整）
        # user_id = self.training_table.item(row, 0).text()  # 假设用户ID存储在第1列

        # 创建并显示训练历史对话框
        dialog = TrainingHistoryDialog(user_id_for_clicked2)  # 传递用户ID给对话框
        dialog.exec_()




    def back_to_login(self):
        self.login_window.show()
        self.close()

#进度环控件类
class ProgressRing(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0  # 0-100
        self.ring_width = 12
        self.colors = {
            'background': QColor(235, 235, 235),
            'progress': QLinearGradient(0, 0, 1, 1),
            'text': QColor(45, 45, 45)
        }
        self.colors['progress'].setColorAt(0, QColor(0, 200, 255))  # 渐变色起始
        self.colors['progress'].setColorAt(1, QColor(0, 120, 255))  # 渐变色结束

    def set_value(self, value):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景环
        rect = QRectF(5, 5, self.width() - 10, self.height() - 10)
        pen = QPen(self.colors['background'], self.ring_width)
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)

        # 绘制进度环
        pen.setBrush(self.colors['progress'])
        pen.setWidth(self.ring_width)
        painter.setPen(pen)
        angle = int(self.value * 3.6 * 16)  # 角度转换
        painter.drawArc(rect, 90 * 16, -angle)

        # 绘制文字
        painter.setFont(QFont('Microsoft YaHei', 16, QFont.Bold))
        painter.setPen(QPen(self.colors['text']))
        painter.drawText(rect, Qt.AlignCenter, f"{self.value}%")

#设置动作强度对话框类
class IntensitySettingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("动作强度设置")
        self.setMinimumSize(500, 400)  # 设定最小尺寸保证布局

        # ================== 全局样式设置 ==================
        self.setStyleSheet("""
            QDialog {
                background: #F7F9FC;
            }
            QLabel {
                color: #2C3E50;
                font-family: 'Segoe UI';
            }
            QGroupBox {
                border: 1px solid #E0E6ED;
                border-radius: 8px;
                margin-top: 20px;
                font: 14px '微软雅黑';
                color: #5E6D82;
            }
            QRadioButton {
                color: #2C3E50;
                font: 14px 'Segoe UI';
                min-height: 40px;
                padding: 8px 0;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6B9EF8, stop:1 #7CABFF);
                color: white;
                border-radius: 6px;
                padding: 12px 30px;
                font: 16px '微软雅黑';
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5C8DE4, stop:1 #6D9BF1);
            }
        """)

        # ================== 主布局 ==================
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # 标题
        title_label = QLabel("选择训练强度")
        title_label.setStyleSheet("""
            font: bold 24px '微软雅黑';
            color: #2C3E50;
            padding-bottom: 10px;
            border-bottom: 2px solid #E0E6ED;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 图标（使用内置图标示例，推荐实际使用SVG文件）
        icon_label = QLabel()
        icon_pixmap = QPixmap("D:/文档/我的图片/动作强度.png").scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon_label.setPixmap(icon_pixmap)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("padding-bottom: 10px;")
        main_layout.addWidget(icon_label)

        # ================== 强度选项卡片 ==================
        self.intensity_group = QGroupBox()
        intensity_layout = QVBoxLayout(self.intensity_group)
        intensity_layout.setSpacing(15)
        intensity_layout.setContentsMargins(20, 5, 20, 5)

        self.radio_buttons = []
        intensities = [
            ("初级强度", "目标角度范围：60° - 90°\n重复次数：5 - 10次\n持续时间：10 - 15秒\n休息时间：15 - 20秒"),
            ("中级强度", "目标角度范围：90° - 120°\n重复次数：10 - 15次\n持续时间：15 - 20秒\n休息时间：10 - 15秒"),
            ("高级强度", "目标角度范围：120° - 150°\n重复次数：15 - 20次\n持续时间：20 - 30秒\n休息时间：5 - 10秒")
        ]

        for intensity, description in intensities:
            item_widget = QWidget()
            item_layout = QVBoxLayout(item_widget)
            item_layout.setSpacing(8)

            # 单选按钮
            radio = QRadioButton(intensity)
            radio.setStyleSheet("""
                QRadioButton::indicator {
                    width: 20px;
                    height: 20px;
                }
                QRadioButton::indicator::checked {
                    background: #6B9EF8;
                    border: 5px solid white;
                    border-radius: 10px;
                }
            """)

            # 描述标签
            desc_label = QLabel(description)
            desc_label.setStyleSheet("""
                color: #8798AD;
                font: 12px 'Microsoft YaHei';
                margin-left: 28px;
                line-height: 1.5;
            """)

            item_layout.addWidget(radio)
            item_layout.addWidget(desc_label)
            intensity_layout.addWidget(item_widget)
            self.radio_buttons.append(radio)

        main_layout.addWidget(self.intensity_group)

        # ================== 保存按钮 ==================
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 20, 0, 0)

        save_btn = QPushButton("保存设置")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self.save_intensity)

        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addStretch()

        main_layout.addWidget(btn_container)

    def save_intensity(self):
        selected_intensity = None
        for i, radio_button in enumerate(self.radio_buttons):
            if radio_button.isChecked():
                selected_intensity = i  # 0: 初级, 1: 中级, 2: 高级
                break

        if selected_intensity is not None:
            # 更新当前任务的强度设置
            QMessageBox.information(self, "保存成功", f"已保存 {['初级', '中级', '高级'][selected_intensity]} 强度设置")
            self.accept()
        else:
            QMessageBox.warning(self, "保存失败", "请选择一个强度级别")

class RehabAssistantDialog(QDialog):
    def __init__(self, user_id, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.messages = []
        self.init_ui()
        self.setWindowTitle("🤖 康复小助手")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

    def init_ui(self):
        self.setFixedSize(1000, 600)
        main_layout = QVBoxLayout(self)

        # 对话展示区
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(20, 20, 20, 20)
        self.scroll_layout.setSpacing(8)
        self.scroll_area.setWidget(self.scroll_content)

        # 创建背景板
        self.background = QWidget()
        self.background.setStyleSheet("background-color: white;")
        self.background_layout = QVBoxLayout(self.background)
        self.background_layout.addWidget(self.scroll_area)

        # 输入区域
        input_layout = QHBoxLayout()
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("💡 Hi~ 我是基于qwen-max大模型的智能康复助手 ✨\n✍️ 输入康复问题（示例：如何正确进行桥式训练？）")
        self.question_input.setFixedHeight(80)
        send_button = QPushButton("发送")
        send_button.setStyleSheet("background-color: #409eff; color: white;")
        send_button.setFixedSize(100, 80)

        # 样式设置
        self.setStyleSheet("""
            QDialog {
                background: #f0f5ff;
                border-radius: 12px;
            }
            QTextEdit {
                font-size: 14px;
                border: 2px solid #b0d5ff;
                border-radius: 8px;
                padding: 10px;
                background: white;
            }
            QPushButton {
                background: #409eff;
                color: black;
                border-radius: 6px;
                padding: 8px;
                font: 500 14px '微软雅黑';
            }
            QPushButton:hover {
                background: #66b1ff;
            }
            QLabel {
                font-size: 14px;
            }
        """)

        input_layout.addWidget(self.question_input)
        input_layout.addWidget(send_button)

        # 预设问题区域
        preset_layout = QGridLayout()
        preset_questions = [
            "如何正确进行桥式训练？",
            "康复训练需要注意哪些事项？",
            "康复期间可以吃什么食物？",
            "康复训练多久可以见效？",
            "康复训练期间感到疼痛怎么办？",
            "康复训练的频率应该是多少？"
        ]

        for i, question in enumerate(preset_questions):
            btn = QPushButton(question)
            btn.setStyleSheet("""
                QPushButton {
                    background: #e9f3ff;
                    color: #409eff;
                    border-radius: 6px;
                    padding: 8px;
                    font: 500 12px '微软雅黑';
                }
                QPushButton:hover {
                    background: #ccebff;
                }
            """)
            btn.clicked.connect(lambda _, q=question: self.send_preset_message(q))
            preset_layout.addWidget(btn, i // 3, i % 3)
            preset_layout.setColumnStretch(i % 3, 1)
            preset_layout.setColumnMinimumWidth(0, 200)  # 每列最小宽度200px

        # 组装布局
        self.background_layout.addLayout(input_layout)
        self.background_layout.addLayout(preset_layout)
        main_layout.addWidget(self.background)

        # 连接信号
        send_button.clicked.connect(self.send_message)
        self.question_input.textChanged.connect(self.check_enter)

    def check_enter(self):
        if "\n" in self.question_input.toPlainText():
            self.send_message()

    def send_message(self):
        question = self.question_input.toPlainText().strip().replace("\n", "")
        if not question:
            return

        self.add_message("user", question)
        self.question_input.clear()

        try:
            response = self.call_bailian_api(question)
            self.add_message("assistant", response)
        except Exception as e:
            self.add_message("system", f"服务暂时不可用: {str(e)}")

    def send_preset_message(self, question):
        self.question_input.setText(question)
        self.send_message()

    def calculate_text_height(self, content):
        # 创建临时文本容器
        test_doc = QTextDocument()
        test_doc.setDefaultFont(QFont("Microsoft YaHei", 14))
        test_doc.setTextWidth(self.scroll_area.width() * 0.6 - 40)  # 扣除边距

        # 设置文本内容（需转换换行符）
        formatted_content = content.replace('\n', '<br>')
        test_doc.setHtml(f"<div style='line-height: 1.8em;'>{formatted_content}</div>")

        # 计算实际高度（含行高）
        return int(test_doc.size().height()) + 20  # 增加安全边距

    def add_message(self, role, content):
        # 创建消息容器
        message_container = QWidget()
        container_layout = QHBoxLayout(message_container)

        # 添加头像
        avatar = QLabel()
        avatar.setFixedSize(40, 40)
        avatar.setStyleSheet("border-radius: 20px;")
        avatar.setAlignment(Qt.AlignCenter)

        if role == "user":
            avatar.setStyleSheet("""
                QLabel {
                    border-radius: 20px;
                }
            """)
            avatar.setToolTip("我")
            container_layout.setAlignment(Qt.AlignRight)
            avatar_path = "D:/文档/我的音乐/患者.png"  # 替换为用户头像的本地路径
        else:
            avatar.setStyleSheet("""
                QLabel {
                    border-radius: 20px;
                }
            """)
            avatar.setToolTip("康复小助手")
            container_layout.setAlignment(Qt.AlignLeft)
            avatar_path = "D:/文档/我的音乐/AI小助手.png"  # 替换为康复小助手头像的本地路径

        # 加载本地图标
        pixmap = QPixmap(avatar_path)
        if not pixmap.isNull():
            avatar.setPixmap(pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            print("无法加载头像图片，请检查路径是否正确")
        message_bubble = QTextEdit(content)
        message_bubble.setReadOnly(True)
        message_bubble.setFrameShape(QFrame.NoFrame)

        # 动态尺寸计算优化
        font_metrics = QFontMetrics(message_bubble.font())

        # 精确计算文本宽度（代替原估算方式）
        text_width = font_metrics.width(content) + 20  # 增加20px安全边距
        max_bubble_width = int(self.scroll_area.width() * 0.6)  # 可视区域60%

        # 智能宽度设置
        bubble_width = min(max(text_width, 60), max_bubble_width)  # 最小60px

        # 智能高度计算
        line_count = 1
        if text_width > max_bubble_width:
            line_count = (text_width // max_bubble_width) + 1
        bubble_height = max(font_metrics.height() * line_count + 10, 30)  # 最小30px

        # 紧凑样式设置
        message_bubble.setStyleSheet(f"""
               QTextEdit {{
                   min-width: {bubble_width}px;
                   max-width: {max_bubble_width}px;
                   min-height: {bubble_height}px;
                   padding: 8px 12px;
                   margin: 2px 5px;
                   line-height: {font_metrics.height() * 0.9}px;  # 紧凑行高
               }}
           """)

        # 布局优化（减少边距）
        container_layout.setContentsMargins(5, 5, 5, 5)  # 原15,8,15,8
        container_layout.setSpacing(8)  # 原10px间距

        # 对齐方式优化
        if role == "user":
            message_bubble.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            container_layout.addWidget(message_bubble)
            container_layout.addWidget(avatar)
        else:
            message_bubble.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            container_layout.addWidget(avatar)
            container_layout.addWidget(message_bubble)

        # 添加到滚动区域
        self.scroll_layout.addWidget(message_container)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def call_bailian_api(self, question):
        api_key = 'sk-39892aafc154458b90c494a37eb7f86e'
        response = dashscope.Generation.call(
            api_key=api_key,
            model="qwen-max",
            messages=[{'role': 'user', 'content': question}],
            result_format='message'
        )
        return response.output.choices[0].message.content

# 患者训练窗口类
class RehabTrainingUI(QMainWindow):
    def __init__(self, login_window, user_id):
        super().__init__()

        self.setWindowTitle("康复训练")
        self.setGeometry(500, 35, 1000, 900)
        self.login_window = login_window
        self.user_id = user_id

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # main_layout = QVBoxLayout(main_widget)
        self.main_layout = QVBoxLayout(main_widget)  # 定义 main_layout 并设置为主窗口的布局
        self.main_layout.setSpacing(5)  # 减少垂直间距

        # 创建顶部标题栏
        title_label = QLabel("🏥 智能康复训练系统")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: 800;
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
            }
        """)
        self.main_layout.addWidget(title_label)
        # ================= 新增进度环区域 =================
        progress_group = QGroupBox()
        progress_layout = QVBoxLayout(progress_group)

        # 创建进度环（直径200px）
        self.progress_ring = ProgressRing()
        self.progress_ring.setFixedSize(80, 80)

        # 添加进度标签
        self.progress_label = QLabel("目标次数：0 | 已完成：0")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 10px; color: #666;")

        progress_layout.addWidget(self.progress_ring, 0, Qt.AlignCenter)
        progress_layout.addWidget(self.progress_label)

        # # 将进度组添加到主布局的顶部区域
        # main_layout.insertWidget(2, progress_group)  # 插入到标题下方
        # 创建一个水平布局，用于容纳用户信息和任务选择区域
        user_task_layout = QHBoxLayout()
        user_task_layout.setSpacing(20)  # 两个区域间的间距建议20-30px

        # 创建用户信息区域的布局，也是水平布局
        user_info_layout = QHBoxLayout()
        # 创建一个QLabel用于显示用户头像，并设置头像图片
        self.user_avatar = QLabel()
        self.user_avatar.setPixmap(QPixmap("D:\\文档\\我的图片\\患者用户1.png").scaled(30, 30, Qt.KeepAspectRatio))  # 调整头像大小，保持宽高比
        # 为头像绑定点击事件
        self.user_avatar.mousePressEvent = self.on_avatar_click
        # 创建一个QLabel用于显示用户名，初始化时显示"用户名: "
        self.user_name_label = QLabel("用户名:")
        # 调用load_user_name()函数加载实际的用户名
        self.load_user_name()
        # 将头像QLabel添加到用户信息布局中
        user_info_layout.addWidget(self.user_avatar)
        # 将用户名QLabel添加到用户信息布局中
        user_info_layout.addWidget(self.user_name_label)
        # 设置用户信息布局中组件之间的间距为1像素
        # user_info_layout.setSpacing(5)
        # user_info_layout.setContentsMargins(0, 0, 0, 0)  # 去除默认边距
        user_info_layout.addStretch()  # 在最后添加弹性空间

        # 创建任务选择区域的布局，同样是水平布局

        task_layout = QHBoxLayout()
        # 创建一个QLabel用于显示任务图标，并设置图标图片
        task_icon = QLabel()
        task_icon.setPixmap(QPixmap("D:\\文档\\我的图片\\任务.png").scaled(30, 30, Qt.KeepAspectRatio))  # 使用本地图片，调整图标大小，保持宽高比
        # 创建一个QLabel用于显示任务选择的标签
        task_label = QLabel("选择任务:")
        # 创建一个QComboBox用于选择任务
        self.task_combo = QComboBox()
        # 加载任务
        self.load_tasks()
        # 设置任务选择下拉框的样式
        self.task_combo.setStyleSheet("""
                            QComboBox {
                                padding: 8px;  # 内部边距
                                border: 2px solid #498db;  # 边框颜色和宽度
                                border-radius: 6px;  # 边框圆角
                                min-width: 250px;  # 最小宽度
                            }
                        """)
        # 将任务图标QLabel添加到任务选择布局中
        task_layout.addWidget(task_icon)
        # 将任务选择标签QLabel添加到任务选择布局中(后添前)
        task_layout.addWidget(task_label)
        # 将任务选择下拉框添加到任务选择布局中
        task_layout.addWidget(self.task_combo)
        task_layout.addWidget(progress_group)  # 将进度环添加到任务选择行尾
        task_layout.setStretchFactor(self.task_combo, 5)  # 任务选择框占3份宽度
        task_layout.setStretchFactor(progress_group, 1)  # 进度环占1份宽度
        task_layout.addStretch()  # 在最后添加弹性空间
        # 将用户信息布局和任务选择布局添加到用户任务组合布局中
        user_task_layout.addLayout(user_info_layout)
        user_task_layout.addLayout(task_layout)
        # 将用户任务组合布局添加到主布局中
        self.main_layout.addLayout(user_task_layout)

        # 创建视频区域
        video_layout = QHBoxLayout()

        # 视频容器样式
        video_style = """
            QLabel {
                border: 3px solid #3498db;
                border-radius: 10px;
                background: #ecf0f1;
            }
        """
        video_style1 = """
            QFrame {
                border: 3px solid #3498db;
                border-radius: 10px;
                background: #ecf0f1;
            }
        """
        # 创建容器框架
        frame = QFrame()
        frame.setStyleSheet(video_style1)
        frame.setFixedSize(350, 250)
        video_layout.addWidget(frame)

        # 示例视频

        self.example_video_widget = QVideoWidget()
        self.example_video_widget.setFixedSize(344, 244)  # 调整大小
        self.example_video_widget.setStyleSheet(video_style1)
        # video_layout.addWidget(self.example_video_widget)

        # 将视频控件放入框架
        layout = QVBoxLayout(frame)
        layout.addWidget(self.example_video_widget)
        layout.setContentsMargins(0, 0, 0, 0)  # 去除默认边距

        # 骨骼检测视频
        self.skeleton_label = QLabel()
        self.skeleton_label.setFixedSize(350, 250)  # 调整大小
        self.skeleton_label.setStyleSheet(video_style)
        video_layout.addWidget(self.skeleton_label)

        self.main_layout.addLayout(video_layout)

        # 创建得分和评价区域
        score_eval_layout = QHBoxLayout()
        score_eval_layout.setContentsMargins(0, 0, 0, 0)  # 移除默认边距
        score_eval_layout.setSpacing(0)  # 移除子布局间距

        # ================= 得分区域 =================
        # 创建一个 QWidget 容器，用于容纳得分区域的所有控件
        score_container = QWidget()
        # 设置得分区域的样式
        score_container.setStyleSheet("""
            background-color: #f5f6fa;  # 背景颜色
            border-radius: 10px;        # 边框圆角
            padding: 15px;              # 内边距
            min-width: 200px;           # 最小宽度，保持对称
        """)
        # 创建一个 QVBoxLayout 布局，用于管理得分区域内的控件排列
        score_container_layout = QVBoxLayout(score_container)
        # 设置布局中控件的对齐方式为居中
        score_container_layout.setAlignment(Qt.AlignCenter)

        # 创建一个 QLabel 用于显示得分图标
        score_icon = QLabel()
        # 设置得分图标的图片，并将其缩放到 40x40 像素，保持宽高比
        score_icon.setPixmap(QPixmap("D:/文档/我的图片/得分.png").scaled(40, 40, Qt.KeepAspectRatio))
        # 将得分图标添加到布局中，并设置其对齐方式为居中
        score_container_layout.addWidget(score_icon, alignment=Qt.AlignCenter)

        # 创建一个 QLabel 用于显示得分文本
        self.score_label = QLabel("得分: 0.0000")
        # 设置得分文本的样式
        self.score_label.setStyleSheet("""
             QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                qproperty-alignment: AlignCenter;
            }
        """)
        # 将得分文本添加到布局中，并设置其对齐方式为居中
        score_container_layout.addWidget(self.score_label, alignment=Qt.AlignCenter)

        # ================= 评价区域 =================
        evaluation_container = QWidget()
        evaluation_container.setStyleSheet("""
            background-color: #f5f6fa;
            border-radius: 10px;
            padding: 15px;
            min-width: 200px;  # 与得分区域保持一致
        """)
        evaluation_container_layout = QVBoxLayout(evaluation_container)
        evaluation_container_layout.setAlignment(Qt.AlignCenter)  # 内容居中

        self.evaluation_icon = QLabel()
        self.evaluation_icon.setPixmap(QPixmap("D:/文档/我的图片/评价.png").scaled(40, 40, Qt.KeepAspectRatio))
        evaluation_container_layout.addWidget(self.evaluation_icon, alignment=Qt.AlignCenter)

        self.evaluation_content = QLabel("请开始动作")
        self.evaluation_content.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                qproperty-alignment: AlignCenter;
            }
        """)
        evaluation_container_layout.addWidget(self.evaluation_content)

        # ================= 组合布局 =================
        score_eval_layout.addWidget(score_container)
        score_eval_layout.addWidget(evaluation_container)

        # 设置两个区域的拉伸比例相同（保证宽度一致）
        score_eval_layout.setStretch(0, 1)
        score_eval_layout.setStretch(1, 1)

        self.main_layout.addLayout(score_eval_layout)

        # 创建动作解析模块的垂直布局容器
        self.action_analysis_layout = QVBoxLayout()

        # 创建标题标签显示"动作解析:"
        self.action_analysis_label = QLabel("动作解析:")

        # 创建内容显示区域，设置默认提示文本
        self.action_analysis_content = QLabel("请选择一个任务以查看动作解析")

        # 禁用文本自动换行，保持内容单行显示（直到手动换行）
        self.action_analysis_content.setWordWrap(False)

        # 将标题和内容标签添加到垂直布局中（按添加顺序从上到下排列）
        self.action_analysis_layout.addWidget(self.action_analysis_label)
        self.action_analysis_layout.addWidget(self.action_analysis_content)

        # 将整个动作解析布局添加到主界面布局中
        self.main_layout.addLayout(self.action_analysis_layout)

        # 主按钮样式
        button_style = """
        QPushButton {
            border-radius: 8px;
            padding: 12px 28px;
            font: 500 14px '微软雅黑';
            transition: all 0.3s ease;
            min-width: 100px;
        }

        QPushButton:hover {
            transform: translateY(-1px);
        }

        QPushButton:pressed {
            transform: translateY(1px);
        }
        """

        # 创建按钮区域
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("🏃️ 开始训练")
        self.start_button.setStyleSheet(f"""
            {button_style}
            QPushButton {{
                background: rgba(41, 128, 185, 0.1);
                border: 1px solid rgba(41, 128, 185, 0.2);
                color: #2C81BA;
            }}
            QPushButton:hover {{
                background: rgba(41, 128, 185, 0.15);
                border-color: #2C81BA;
            }}
        """)
        self.start_button.setCursor(Qt.PointingHandCursor)

        # 辅助按钮样式（训练历史）
        self.history_button = QPushButton("📅 训练历史")
        self.history_button.setStyleSheet(f"""
            {button_style}
            QPushButton {{
                background: rgba(41, 128, 185, 0.1);
                border: 1px solid rgba(41, 128, 185, 0.2);
                color: #2C81BA;
            }}
            QPushButton:hover {{
                background: rgba(41, 128, 185, 0.15);
                border-color: #2C81BA;
            }}
        """)
        self.history_button.setCursor(Qt.PointingHandCursor)

        # 设置按钮（强调色）
        self.setting_button = QPushButton("⚙️ 动作设置")
        self.setting_button.setStyleSheet(f"""
            {button_style}
            QPushButton {{
                background: rgba(41, 128, 185, 0.1);
                border: 1px solid rgba(41, 128, 185, 0.2);
                color: #2C81BA;
            }}
            QPushButton:hover {{
                background: rgba(41, 128, 185, 0.15);
                border-color: #2C81BA;
            }}
        """)
        self.setting_button.setCursor(Qt.PointingHandCursor)

        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.history_button)
        self.button_layout.addWidget(self.setting_button)
        self.main_layout.addLayout(self.button_layout)


        # 创建返回按钮
        self.back_button = QPushButton("返回登录")
        self.back_button.setStyleSheet("background-color: #2196F3; color: white; font-size: 14px;")
        self.back_button.clicked.connect(self.back_to_login)
        # self.main_layout.addWidget(self.back_button)

        # 初始化OpenPose
        params = {
            "model_folder": "../../../models/",
            "hand": False,
            "number_people_max": 1,
            "disable_blending": False,
            "net_resolution": "320x176"
        }
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        # 初始化系统组件
        self.voice_assistant = VoiceAssistant()
        self.rehab_counter = RehabilitationCounter()

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "摄像头错误", "无法打开摄像头，请检查摄像头是否可用。")
            return

        # 初始化定时器
        self.timer = QTimer()

        # 绑定触发函数
        self.timer.timeout.connect(self.update_frame)

        # 当调用 self.timer.start(30) 时，会启动这个定时器，每30毫秒调用一次 update_frame 方法。
        self.timer.start(30)

        # 当前任务配置
        self.current_task_config = None

        # 连接康复训练师界面的任务更新信号到刷新任务方法(弃用)
        # self.login_window.therapist_window.task_updated.connect(self.load_tasks)

        # 添加布尔变量控制检测状态
        self.is_detecting = False

        # 设置开始按钮的点击事件
        self.start_button.clicked.connect(self.toggle_detection)

        # 设置训练历史按钮的点击事件
        self.history_button.clicked.connect(self.show_training_history)

        # 设置动作设置按钮的点击事件
        self.setting_button.clicked.connect(self.show_intensity_setting)

        # 创建康复小助手按钮
        self.rehab_assistant_button = QPushButton("🤖 康复小助手")
        self.rehab_assistant_button.setStyleSheet(f"""
                            {button_style}
                            QPushButton {{
                                background: rgba(41, 128, 185, 0.1);
                                border: 1px solid rgba(41, 128, 185, 0.2);
                                color: #2C81BA;
                            }}
                            QPushButton:hover {{
                                background: rgba(41, 128, 185, 0.15);
                                border-color: #2C81BA;
                            }}
                        """)
        self.rehab_assistant_button.clicked.connect(self.toggle_rehab_assistant)
        self.button_layout.addWidget(self.rehab_assistant_button)

        # # 创建智能输入框
        # self.question_input = QTextEdit()
        # self.question_input.setPlaceholderText("💡 Hi~ 我是基于qwen-max大模型的智能康复助手 ✨\n✍️ 输入康复问题（示例：如何正确进行桥式训练？）")
        # self.question_input.setStyleSheet("""
        #     QTextEdit {
        #         font-size: 18px;
        #         color: #2a5b87;
        #         border: 2px solid #b0d5ff;
        #         border-radius: 12px;
        #         padding: 15px;
        #         margin: 20px 0;
        #         background: #f8fbff;
        #     }
        #     QTextEdit::placeholder {
        #         color: #9ab9d6;
        #         font-style: italic;
        #     }
        # """)
        # self.main_layout.addWidget(self.question_input)

        # # 绑定智能输入检测
        # self.question_input.textChanged.connect(self.handle_text_changed)

        # 创建对话展示区
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid #e3f0ff;
                border-radius: 12px;
                background: #ffffff;
            }
            QScrollBar:vertical {
                width: 10px;
                background: #f0f7ff;
            }
        """)

        # # 创建动态内容容器
        # self.scroll_content = QWidget()
        # self.scroll_content.setStyleSheet("background: #fcfdff;")
        # self.scroll_layout = QVBoxLayout(self.scroll_content)
        # self.scroll_layout.setContentsMargins(20, 20, 20, 20)
        # self.scroll_layout.setSpacing(15)
        #
        # # 智能思考指示器
        # self.model_output_label = QLabel("⏳ 深度思考" + " " * 30)
        # self.model_output_label.setStyleSheet("""
        #     QLabel {
        #         font-size: 18px;
        #         color: #409eff;
        #         font-weight: 500;
        #         padding: 12px 20px;
        #         background: #f0f8ff;
        #         border-radius: 8px;
        #         border: 1px solid #d4ebff;
        #         animation: dotPulse 1.5s infinite;
        #     }
        #     @keyframes dotPulse {
        #         0% { color: #409eff; }
        #         50% { color: #9ac8ff; }
        #         100% { color: #409eff; }
        #     }
        # """)
        # self.model_output_label.setAlignment(Qt.AlignCenter)
        # self.scroll_layout.addWidget(self.model_output_label)

        # # 设置滚动区域的部件
        # self.scroll_area.setWidget(self.scroll_content)
        # self.main_layout.addWidget(self.scroll_area)
        #
        # # 初始化对话历史
        # self.messages = []

        # # 默认隐藏输入框和输出标签
        # self.question_input.hide()
        # self.scroll_area.hide()

        # # 添加一个标志，用于判断是否按下回车键
        # self.enter_pressed = False
        #
        # # 创建音乐区域
        self.create_music_player()

        # self.action_analysis_content()

    #个人信息点击事件函数
    def on_avatar_click(self,event):
        # 打开个人信息界面
        print("Avatar clicked!")
        self.user_profile_window = UserProfileWindow(self.user_id, self)
        self.user_profile_window.show()

    #创建音乐播放器
    def create_music_player(self):
        # 创建音乐播放器
        self.music_player = QMediaPlayer()

        # 创建音乐区域的主布局
        music_layout = QVBoxLayout()

        # 创建音乐列表按钮
        self.song_list_btn = QPushButton()
        self.song_list_btn.setIcon(QIcon("D:/文档/我的图片/音乐.png"))  # 使用用户提供的箭头图标
        self.song_list_btn.setIconSize(QSize(24, 24))
        self.song_list_btn.setFixedSize(32, 32)
        self.song_list_btn.setStyleSheet("""
               QPushButton {
                   background: transparent;
                   border: none;
                   padding: 4px;
               }
               QPushButton:hover {
                   background: #f0f0f0;
                   border-radius: 4px;
               }
           """)

        # 创建歌曲目录弹窗
        self.song_menu = QMenu()
        self.song_menu.setStyleSheet("""
               QMenu {
                   background: white;
                   border: 1px solid #ddd;
                   padding: 8px;
                   min-width: 180px;
               }
               QListWidget {
                   border: none;
                   outline: none;
               }
               QListWidget::item {
                   padding: 6px 12px;
               }
               QListWidget::item:hover {
                   background: #f5f5f5;
               }
               QListWidget::item:selected {
                   background: #e0e0e0;
               }
           """)

        # 添加歌曲列表
        self.song_list = QListWidget()
        self.song_list.addItems([
            "自然雨声", "梦中的婚礼",
            "菊次郎的夏天", "MySoul",
            "Letting Go", "StartBoy"
        ])
        self.song_list.itemClicked.connect(self.load_selected_music)

        # 将列表嵌入菜单
        song_action = QWidgetAction(self.song_menu)
        song_action.setDefaultWidget(self.song_list)
        self.song_menu.addAction(song_action)

        # 连接按钮点击事件
        self.song_list_btn.setMenu(self.song_menu)
        self.song_list_btn.setStyleSheet("QPushButton::menu-indicator { width:0px; }")


        # 创建播放控制按钮布局
        control_layout = QHBoxLayout()

        # 创建循环模式按钮（带弹出菜单）
        self.loop_button = QPushButton()
        self.loop_button.setIcon(QIcon("D:/文档/我的图片/列表循环.png"))  # 默认列表循环图标
        self.loop_button.setToolTip("播放模式")

        # 创建模式菜单
        self.loop_menu = QMenu()
        self.loop_menu.setStyleSheet("""
                QMenu {
                    background: white;
                    border: 1px solid #ddd;
                    padding:5px;
                }
                QMenu::item {
                    padding: 5px 20px 5px 10px;
                }
                QMenu::item:selected { 
                    background: #f0f0f0;
                    display: none;
                }
            """)

        # 添加模式选项
        self.mode_actions = {
            "list_loop": self.loop_menu.addAction(
                QIcon("D:/文档/我的图片/列表循环.png"), "列表循环"),
            "single_loop": self.loop_menu.addAction(
                QIcon("D:/文档/我的图片/循环.png"), "单曲循环"),
            "random_play": self.loop_menu.addAction(
                QIcon("D:/文档/我的图片/随机播放.png"), "随机播放")
        }

        # 连接信号
        self.loop_button.setMenu(self.loop_menu)
        self.loop_menu.triggered.connect(self.update_play_mode)

        control_layout.addWidget(self.loop_button)
        self.current_play_mode = "list_loop"  # 初始模式

        # 添加上一曲按钮
        self.prev_button = QPushButton()
        self.prev_button.setIcon(QIcon("D:/文档/我的图片/上一曲.png"))
        self.prev_button.clicked.connect(self.play_previous_music)  # 新增事件绑定
        control_layout.addWidget(self.prev_button)

        # 创建播放/暂停切换按钮
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(QIcon("D:/文档/我的图片/播放.png"))  # 初始播放图标
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        control_layout.addWidget(self.play_pause_button)

        # 添加下一曲按钮
        self.next_button = QPushButton()
        self.next_button.setIcon(QIcon("D:/文档/我的图片/下一曲.png"))
        self.next_button.clicked.connect(self.play_next_music)  # 新增事件绑定
        control_layout.addWidget(self.next_button)

        # 创建音量按钮
        self.volume_button = QPushButton()
        self.volume_button.setIcon(QIcon("D:/文档/我的图片/音量.png"))
        self.volume_button.setFixedSize(32, 32)
        self.volume_button.setIconSize(QSize(24, 24))

        # 创建悬浮音量面板
        self.volume_panel = QWidget()
        self.volume_panel.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.volume_panel.setFixedSize(56, 160)
        self.volume_panel.setStyleSheet("""
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            """)

        # 音量面板布局
        volume_layout = QVBoxLayout()
        volume_layout.setContentsMargins(8, 16, 8, 16)

        # 音量加号按钮
        self.vol_add = QPushButton("+")
        self.vol_add.setFixedSize(24, 24)
        self.vol_add.setStyleSheet("""
                QPushButton {
                    background: #4CAF50;
                    color: white;
                    border-radius: 12px;
                    font: bold 14px;
                }
            """)
        self.vol_add.clicked.connect(lambda: self.volume_slider.setValue(100))

        # 垂直音量条
        self.volume_slider = QSlider(Qt.Vertical)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setStyleSheet("""
                QSlider::groove:vertical {
                    width: 4px;
                    background: #E0E0E0;
                    border-radius: 2px;
                }
                QSlider::sub-page:vertical {
                    background: #4CAF50;
                    border-radius: 2px;
                }
                QSlider::handle:vertical {
                    height: 12px;
                    background: #4CAF50;
                    margin: 0 -4px;
                    border-radius: 6px;
                }
            """)

        # 音量百分比显示
        self.vol_percent = QLabel("100%")
        self.vol_percent.setAlignment(Qt.AlignCenter)
        self.vol_percent.setStyleSheet("font: 12px; color: #666;")

        # 静音按钮
        self.mute_btn = QPushButton()
        self.mute_btn.setIcon(QIcon("D:/文档/我的图片/静音.png"))
        self.mute_btn.setIconSize(QSize(20, 20))
        self.mute_btn.clicked.connect(self.toggle_mute)

        control_layout.addWidget(self.song_list_btn)

        # 组装布局
        volume_layout.addWidget(self.vol_add, 0, Qt.AlignHCenter)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.vol_percent)
        volume_layout.addWidget(self.mute_btn, 0, Qt.AlignHCenter)
        self.volume_panel.setLayout(volume_layout)

        # 连接信号
        self.volume_button.clicked.connect(self.show_volume_panel)
        self.volume_slider.valueChanged.connect(self.update_volume_display)

        # 创建带滑块的弹出菜单
        volume_menu = QMenu()
        volume_action = QWidgetAction(volume_menu)
        volume_action.setDefaultWidget(self.volume_slider)
        volume_menu.addAction(volume_action)
        self.volume_button.setMenu(volume_menu)
        self.volume_button.setStyleSheet("QPushButton::menu-indicator { width: 0px; }")
        control_layout.addWidget(self.volume_button)

        # # 添加进度条
        # self.progress_slider = QSlider(Qt.Horizontal)
        # self.progress_slider.setRange(0, 100)
        # self.progress_slider.setValue(0)
        # self.progress_slider.sliderMoved.connect(self.set_music_position)
        music_layout.addLayout(control_layout)
        # music_layout.addWidget(self.progress_slider)

        # 初始化播放状态跟踪
        self.music_player.stateChanged.connect(self.update_play_icon)
        self.volume_slider.valueChanged.connect(self.music_player.setVolume)

        # 创建音乐区域容器
        music_container = QWidget()
        music_container.setLayout(music_layout)
        # 在create_music_player方法中添加样式设置
        music_container.setStyleSheet("""
            QWidget {
                background: white;  /* 整体背景白色 */
                border: none;
            }
            QPushButton {
                background: white;  /* 按钮背景白色 */
                border: 1px solid white;  /* 边框白色 */
                border-radius: 3px;
                padding: 5px;
                margin: 0px;  /* 去除按钮间距 */
            }
            QPushButton:hover {
                background: #f5f5f5;  /* 悬停微调 */
            }
            QSlider::groove:horizontal {
                background: white;  /* 进度条背景白色 */
                border: 1px solid #ddd;
            }
            QComboBox {
                border: 1px solid #ddd;  /* 下拉框保持浅灰边框 */
                background: white;
            }
            QPushButton::menu-indicator {
                image: none; /* 隐藏下拉箭头 */
           }
        """)

        # 初始化音乐文件路径
        self.music_files = {
            "自然雨声": "D:/文档/我的音乐/light_music1.mp3",
            "梦中的婚礼": "D:/文档/我的音乐/light_music2.mp3",
            "菊次郎的夏天": "D:/文档/我的音乐/light_music3.mp3",
            "MySoul": "D:/文档/我的音乐/light_music4.mp3",
            "Letting Go": "D:/文档/我的音乐/LettingGo.mp3",
            "StartBoy" : "D:/文档/我的音乐/StartBoy.mp3",
        }

        # 连接播放器信号
        # self.music_player.positionChanged.connect(self.update_progress)
        # self.music_player.durationChanged.connect(self.update_duration)
        self.main_layout.addWidget(music_container)

        # 初始化音乐列表（新增以下代码）
        self.music_tracks = [
            "自然雨声",
            "梦中的婚礼",
            "菊次郎的夏天",
            "MySoul",
            "Letting Go",
            "StartBoy"
        ]
        self.current_track_index = -1  # 当前播放索引

        # 设置按钮尺寸（单位：像素）
        BUTTON_SIZE = 48  # 根据实际显示效果调整

        # 为所有控制按钮设置统一样式
        for button in [self.loop_button, self.prev_button,
                       self.play_pause_button, self.next_button,
                       self.volume_button]:
            button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE)  # 固定按钮尺寸
            button.setIconSize(QSize(int(BUTTON_SIZE * 0.6), int(BUTTON_SIZE * 0.6)))  # 图标占按钮60%面积
            button.setStyleSheet("""
                QPushButton {
                    border: none;
                    background: white;
                    padding: 0px;
                }
                QPushButton:hover {
                    background: #f5f5f5;
                }
            """)

        # 特别放大播放按钮（比其他按钮大20%）
        self.play_pause_button.setFixedSize(int(BUTTON_SIZE * 1.2), int(BUTTON_SIZE * 1.2))
        self.play_pause_button.setIconSize(QSize(int(BUTTON_SIZE * 0.8), int(BUTTON_SIZE * 0.8)))

        # 调整布局间距（保持图标间距均匀）
        control_layout.setSpacing(6)  # 6px间距模拟原图效果

        # 在初始化时连接信号
        self.music_player.mediaStatusChanged.connect(self.media_status_handler)

    def show_volume_panel(self):
        pos = self.volume_button.mapToGlobal(QPoint(0, -self.volume_panel.height()))
        self.volume_panel.move(pos)
        self.volume_panel.show()

    def update_volume_display(self, value):
        self.vol_percent.setText(f"{value}%")
        self.music_player.setVolume(value)
        if value == 0:
            self.mute_btn.setIcon(QIcon("D:/文档/我的图片/静音.png"))
        else:
            self.mute_btn.setIcon(QIcon("D:/文档/我的图片/音量开.png"))

    def toggle_mute(self):
        current = self.volume_slider.value()
        if current > 0:
            self.last_volume = current
            self.volume_slider.setValue(0)
        else:
            self.volume_slider.setValue(self.last_volume if hasattr(self, 'last_volume') else 50)


    def update_play_mode(self, action):
        """ 更新播放模式和图标 """
        mode_mapping = {
            self.mode_actions["list_loop"]: ("list_loop", "列表循环"),
            self.mode_actions["single_loop"]: ("single_loop", "单曲循环"),
            self.mode_actions["random_play"]: ("random_play", "随机播放")
        }

        new_mode, tip_text = mode_mapping[action]
        self.current_play_mode = new_mode
        self.loop_button.setIcon(action.icon())
        self.loop_button.setToolTip(tip_text)

        # 实现实际播放逻辑
        if self.music_player.state() == QMediaPlayer.PlayingState:
            current_position = self.music_player.position()
            self.music_player.stop()
            self.music_player.setPosition(current_position)  # 保持播放进度
            self.music_player.play()

    def play_next_music(self):
        """ 下一曲功能实现 """
        if not self.music_tracks:
            return

        # 模式判断逻辑
        if self.current_play_mode == "random_play":
            new_index = random.randint(0, len(self.music_tracks) - 1)
        elif self.current_play_mode == "single_loop":
            new_index = self.current_track_index
        else:  # 默认列表循环
            new_index = (self.current_track_index + 1) % len(self.music_tracks)

        # 更新列表选中项
        self.song_list.setCurrentRow(new_index)

        # 加载并播放
        self.load_selected_music(self.song_list.item(new_index))
        self.current_track_index = new_index

        if self.music_player.state() != QMediaPlayer.PlayingState:
            self.music_player.play()

    def media_status_handler(self, status):
        """ 处理播放结束事件 """
        if status == QMediaPlayer.EndOfMedia:
            if self.current_play_mode == "single_loop":
                self.music_player.play()
            else:
                self.play_next_music()

    def play_previous_music(self):
        """ 上一曲功能实现（支持多种播放模式）"""
        if not self.music_tracks:
            QMessageBox.warning(self, "播放错误", "当前没有可播放的曲目")
            return

        try:
            # 根据播放模式处理索引
            if self.current_play_mode == "random_play":
                # 随机模式：生成新随机索引（排除当前）
                new_index = random.choice([
                    i for i in range(len(self.music_tracks))
                    if i != self.current_track_index
                ])
            elif self.current_play_mode == "single_loop":
                # 单曲循环：保持当前索引
                new_index = self.current_track_index
            else:
                # 列表模式：循环递减
                new_index = (self.current_track_index - 1) % len(self.music_tracks)

            # 验证索引有效性
            if not 0 <= new_index < len(self.music_tracks):
                raise IndexError("无效的曲目索引")

            # 更新UI和状态
            self.song_list.setCurrentRow(new_index)
            current_item = self.song_list.currentItem()

            if current_item is None:
                raise ValueError("未找到对应的曲目")

            # 加载并播放
            self.load_selected_music(current_item)
            self.current_track_index = new_index

            # 自动继续播放（如果之前是播放状态）
            if self.music_player.state() != QMediaPlayer.PausedState:
                self.music_player.play()

        except Exception as e:
            QMessageBox.critical(self, "播放错误", f"切换上一曲失败: {str(e)}")
            print(f"[ERROR] 上一曲切换异常: {traceback.format_exc()}")

    def toggle_play_pause(self):
        if self.music_player.state() == QMediaPlayer.PlayingState:
            self.music_player.pause()
        else:
            if self.music_player.mediaStatus() == QMediaPlayer.NoMedia:
                self.load_selected_music()
            self.music_player.play()

    def update_play_icon(self, state):
        self.play_pause_button.setIcon(
            QIcon("D:/文档/我的图片/暂停.png" if state == QMediaPlayer.PlayingState
                  else "D:/文档/我的图片/播放.png")
        )

    def load_selected_music(self,item):
        """ 修改后的加载方法（添加索引跟踪）"""
        selected_music = item.text()
        if selected_music != "选择音乐":
            try:
                # 更新当前播放索引
                self.current_track_index = self.music_tracks.index(selected_music)
            except ValueError:
                self.current_track_index = -1
                return

            # 剩余原有加载逻辑...
            music_path = self.music_files.get(selected_music, "")
            if os.path.exists(music_path):
                self.music_player.setMedia(QMediaContent(QUrl.fromLocalFile(music_path)))
            else:
                print(f"音乐文件不存在: {music_path}")


    # def set_music_position(self, position):
    #     """ 兼容旧版本代码的过渡方法 """
    #     if self.music_player.duration() > 0:
    #         self.music_player.setPosition(int(position * self.music_player.duration() / 100))

    # def update_duration(self, duration):
    #     """ 更新进度条最大时长 """
    #     if duration > 0:
    #         # 将进度条范围设为0-100（百分比模式）
    #         self.progress_slider.setMaximum(100)
    #     else:
    #         self.progress_slider.setMaximum(0)
    #
    # def update_progress(self, position):
    #     """ 更新播放进度 """
    #     if self.music_player.duration() > 0:
    #         # 将位置转换为百分比
    #         progress = int((position / self.music_player.duration()) * 100)
    #         self.progress_slider.setValue(progress)

    # def toggle_rehab_assistant(self):
    #     # 切换康复小助手的显示状态
    #     if self.question_input.isVisible():
    #         self.question_input.hide()
    #         self.scroll_area.hide()
    #         self.rehab_assistant_button.setText("🤖 开启康复小助手")
    #     else:
    #         self.question_input.show()
    #         self.scroll_area.show()
    #         self.rehab_assistant_button.setText("🤖 关闭康复小助手")

    def toggle_rehab_assistant(self):
        dialog = RehabAssistantDialog(self.user_id, self)
        dialog.exec_()

    # def handle_text_changed(self):
    #     # 检查用户是否按下回车键
    #     if self.enter_pressed:
    #         self.enter_pressed = False
    #         return
    #
    #     text = self.question_input.toPlainText()
    #     if text.endswith('\n'):
    #         self.enter_pressed = True
    #         self.show_rehab_assistant()

    # def show_rehab_assistant(self):
    #     # 获取用户输入的问题
    #     user_question = self.question_input.toPlainText().strip()
    #     if not user_question:
    #         return
    #
    #     # 添加用户问题到对话历史
    #     self.messages.append({'role': 'user', 'content': user_question})
    #
    #     try:
    #         # 调用模型API
    #         result = self.call_bailian_api()
    #         # 更新界面显示结果
    #         self.model_output_label.setText(f"🤖康复小助手: {result}")
    #     except Exception as e:
    #         self.model_output_label.setText(f"模型调用失败: {str(e)}")
    #
    #     # 清空输入框
    #     self.question_input.clear()

    def call_bailian_api(self):
        # 配置API Key
        api_key = 'sk-39892aafc154458b90c494a37eb7f86e'  # 替换为您的API Key

        try:
            # 调用百炼API
            response = dashscope.Generation.call(
                api_key=api_key,
                model="qwen-max",  # 模型名称，根据您的需求替换
                messages=self.messages,
                result_format='message'  # 结果格式
            )

            # 获取模型的回答
            model_response = response.output.choices[0].message.content

            # 添加模型回答到对话历史
            self.messages.append({'role': 'assistant', 'content': model_response})

            return model_response
        except Exception as e:
            return f"发生错误: {str(e)}"




    def show_intensity_setting(self):
        # 创建一个强度设置对话框实例，传入父窗口self
        dialog = IntensitySettingDialog(self)

        # 显示对话框并等待用户操作，如果用户点击了“确定”按钮，则exec_()返回QDialog.Accepted
        if dialog.exec_() == QDialog.Accepted:
            # 初始化selected_intensity变量为None，用于存储用户选择的强度级别索引
            selected_intensity = None

            # 遍历dialog.radio_buttons列表中的每个单选按钮及其索引i
            for i, radio_button in enumerate(dialog.radio_buttons):
                # 检查当前单选按钮是否被选中
                if radio_button.isChecked():
                    # 将selected_intensity设置为当前单选按钮的索引值
                    # 0对应初级，1对应中级，2对应高级
                    selected_intensity = i
                    # 找到选中的单选按钮后跳出循环
                    break

            # 如果用户确实选择了某个强度级别（即selected_intensity不是None）
            if selected_intensity is not None:
                # 定义不同强度级别的具体配置参数
                intensity_settings = [
                    # 初级难度配置：目标角度范围60-90度，重复次数5-10次，持续时间10-15秒，休息时间15-20秒
                    {"target_angle_range": (60, 90), "target_reps": (5, 10), "duration": (10, 15),
                     "rest_time": (15, 20)},
                    # 中级难度配置：目标角度范围90-120度，重复次数10-15次，持续时间15-20秒，休息时间10-15秒
                    {"target_angle_range": (90, 120), "target_reps": (10, 15), "duration": (15, 20),
                     "rest_time": (10, 15)},
                    # 高级难度配置：目标角度范围120-150度，重复次数15-20次，持续时间20-30秒，休息时间5-10秒
                    {"target_angle_range": (120, 150), "target_reps": (15, 20), "duration": (20, 30),
                     "rest_time": (5, 10)}
                ]

                # 根据用户选择的强度级别更新当前任务配置
                self.current_task_config["target_angle_range"] = intensity_settings[selected_intensity][
                    "target_angle_range"]
                print("任务强度配置之后")
                print(self.current_task_config["targe_angle_range"])

                # 更新语音提示信息，其中{}是一个占位符，将在实际使用时替换为具体的重复次数
                self.current_task_config["voice_prompts"]["rep_complete"] = f"已经成功完成{{}}次标准动作！继续加油！"

    def update_training_progress(self):
        """从数据库获取当前任务的进度"""

        # 获取当前选中的任务文本
        task_text = self.task_combo.currentText()
        if not task_text:
            return  # 如果没有选中任何任务，则直接返回

        # 解析任务信息（示例："手臂屈伸 (目标次数: 20, 截止日期: 2025-05-01, 状态: In Progress)"）
        parts = task_text.split("目标次数: ")
        if len(parts) > 1:
            target_reps = int(parts[1].split(",")[0])  # 提取目标次数并转换为整数
            # task_name = parts[0].strip()  # 提取任务名称并去除前后空格

            # 连接到SQLite数据库以查询已完成次数
            with sqlite3.connect('rehab.db') as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t.completed_reps
                    FROM RehabTasks t
                    JOIN Users u ON t.patient_id = u.user_id
                    WHERE u.user_id = ?
                """, (self.user_id,))  # 假设user_id是当前用户的ID
                result = cursor.fetchone()  # 获取查询结果
                completed = result[0] if result else 0  # 如果有结果则提取已完成次数，否则设为0
                print("获取查询结果")
                print(result)
                print(completed)
                print(target_reps)

            # 计算进度百分比
            progress = int((completed / target_reps) * 100) if target_reps > 0 else 0

            # 更新进度环显示的值和标签上的文本
            self.progress_ring.set_value(progress)
            self.progress_label.setText(
                f"目标次数：{target_reps} | 已完成：{completed} ({progress}%)"
            )

            # 根据进度动态调整颜色
            if progress >= 100:
                self.progress_ring.colors['progress'].setColorAt(0, QColor(0, 255, 127))
                self.progress_ring.colors['progress'].setColorAt(1, QColor(0, 200, 0))
            elif progress >= 75:
                self.progress_ring.colors['progress'].setColorAt(0, QColor(255, 193, 7))
                self.progress_ring.colors['progress'].setColorAt(1, QColor(255, 152, 0))

    def load_user_name(self):
        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM Users WHERE user_id = ?", (self.user_id,))
        user = cursor.fetchone()
        conn.close()
        if user:
            self.user_name_label.setText(f"用户名: {user[0]}")

    def load_tasks(self):
        # 连接到 SQLite 数据库 rehab.db
        conn = sqlite3.connect('rehab.db')
        # 创建一个游标对象，用于执行 SQL 查询
        cursor = conn.cursor()
        # 执行 SQL 查询，获取当前用户未完成的任务
        # 查询的条件是 patient_id 等于当前用户的 ID，并且任务状态不是 'Completed'
        cursor.execute("""
            SELECT task_id, task_description, target_reps, due_date, status 
            FROM RehabTasks 
            WHERE patient_id = ? AND status != 'Completed'
        """, (self.user_id,))
        # 获取查询结果
        tasks = cursor.fetchall()
        # 关闭数据库连接
        conn.close()
        # 清空下拉框中的所有选项
        self.task_combo.clear()
        # 遍历查询结果，将每个任务的描述添加到下拉框中
        for task in tasks:
            # 解包任务信息
            task_id, task_desc, target_reps, due_date, status = task
            # 构造任务描述字符串
            task_text = f"{task_desc} (目标次数: {target_reps}, 截止日期: {due_date}, 状态: {status})"
            # 将任务描述添加到下拉框中
            self.task_combo.addItem(task_text)
        # 如果没有找到任何任务，显示提示信息
        if not tasks:
            # 弹出一个消息框，提示用户当前没有可加载的任务
            QMessageBox.information(self, "提示", "当前没有可加载的任务。")

    def on_state_changed(self, state):
        if state == QMediaPlayer.StoppedState:  # 检测到播放停止
            self.media_player.play()  # 重新播放

    def load_example_video(self):
        """根据选择的任务更新视频"""
        selected_task = self.task_combo.currentText().split(" (")[0]
        task_config = TASK_CONFIG.get(selected_task, {})
        # self.load_example_video(task_config["video_path"])
        video_path = task_config["video_path"]
        # video_path = r"D:\文档\我的图片\V1.mp4"  # 使用绝对路径
        if not os.path.exists(video_path):
            QMessageBox.critical(self, "文件缺失", f"视频文件未找到：{video_path}")
            return

        self.media_player = QMediaPlayer()
        print('1')
        self.media_player.stateChanged.connect(self.on_state_changed)  # 连接状态改变信号

        print('2')

        self.media_player.setVideoOutput(self.example_video_widget)

        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))

        # 开始播放
        self.media_player.play()

        # 检查媒体状态
        if self.media_player.mediaStatus() == QMediaPlayer.InvalidMedia:
            QMessageBox.critical(self, "无效媒体", "视频文件无效或无法播放。")
            return

        # 检查播放器错误
        if self.media_player.error() != QMediaPlayer.NoError:
            error_message = self.media_player.errorString()
            print(f"视频加载失败: {error_message}")
            QMessageBox.warning(self, "视频加载失败", f"无法加载视频: {error_message}")
            return

        print("视频加载成功")

    def back_to_login(self):
        self.save_training_result()
        self.timer.stop()
        self.cap.release()
        self.opWrapper.stop()
        self.login_window.show()
        self.close()

    def set_current_task_config(self):
        # 获取任务选择下拉框中当前选中的任务描述
        selected_task = self.task_combo.currentText()
        # 如果没有选中的任务描述，直接返回
        if not selected_task:
            return
        # 从选中的任务描述中提取任务名称，任务描述的格式为 "任务名称 (其他信息)"
        task_name = selected_task.split(" (")[0]
        # 判断任务名称是否存在于任务配置字典 TASK_CONFIG 中
        if task_name in TASK_CONFIG:
            # 如果存在，直接从 TASK_CONFIG 中获取该任务的配置信息
            self.current_task_config = TASK_CONFIG[task_name]
            print(task_name)
        else:
            # 如果不存在，设置一个默认的任务配置信息
            self.current_task_config = {
                "target_angle_range": (80, 120),  # 目标角度范围
                "voice_prompts": {  # 语音提示内容
                    "too_low": "手臂再抬高一点，加油！",
                    "good_posture": "动作非常标准，继续保持！",
                    "overbend": "请注意不要过度弯曲哦!",
                    "low_confidence": "请将右臂保持在画面中",
                    "rep_complete": "已经成功完成{}次标准动作！继续加油！"
                },
                "joint_ids": {  # 关节点ID
                    "shoulder": 2,  # 肩部
                    "elbow": 3,  # 肘部
                    "wrist": 4  # 腕部
                }
            }

        # 调用 update_action_analysis 方法更新动作解析区域的内容
        self.update_action_analysis()

    # 更新动作解析内容
    def update_action_analysis(self):
        if self.current_task_config:
            task_name = self.task_combo.currentText().split(" (")[0]

            print('任务名称是：')
            print(task_name)

            task_analysis = {
                "Bobath握手练习": {
                    "standard_practice": f"<p><b>{task_name}标准做法：</b></p><ol><li>双手交叉握手，健侧手带动患侧手</li><li>缓慢向上举起</li><li>向一侧耳朵靠近</li></ol>",
                    "precautions": "<p><b style='color:#c0392b;'>❗ 重要注意事项：</b></p><ul><li>避免用力过猛</li><li>保持动作缓慢</li><li style='color:#c0392b;font-weight:800;'>如感到疼痛立即停止！</li></ul>"
                },
                "桥式运动": {
                    "standard_practice": f"<p><b>{task_name}标准做法：</b></p><ol><li>仰卧位，双膝弯曲</li><li>双脚平放在地面上</li><li>缓慢抬起臀部</li></ol>",
                    "precautions": "<p><b style='color:#c0392b;'>❗ 重要注意事项：</b></p><ul><li>避免突然发力</li><li>保持呼吸节奏</li><li style='color:#c0392b;font-weight:800;'>如感到腰部疼痛立即停止！</li></ul>"
                },
                "关节活动度训练": {
                    "standard_practice": f"<p><b>{task_name}标准做法：</b></p><ol><li>缓慢活动关节</li><li>逐渐增加活动范围</li></ol>",
                    "precautions": "<p><b style='color:#c0392b;'>❗ 重要注意事项：</b></p><ul><li>避免过度活动</li><li>保持动作缓慢</li><li style='color:#c0392b;font-weight:800;'>如感到关节疼痛立即停止！</li></ul>"
                },
                "坐位神经滑动练习": {
                    "standard_practice": f"<p><b>{task_name}标准做法：</b></p><ol><li>坐位</li><li>缓慢向前弯腰</li><li>拉伸腰部神经</li></ol>",
                    "precautions": "<p><b style='color:#c0392b;'>❗ 重要注意事项：</b></p><ul><li>避免用力过猛</li><li>保持动作缓慢</li><li style='color:#c0392b;font-weight:800;'>如感到神经压迫立即停止！</li></ul>"
                },
                "伸展运动": {
                    "standard_practice": f"<p><b>{task_name}标准做法：</b></p><ol><li>全身伸展</li><li>重点在四肢和躯干</li></ol>",
                    "precautions": "<p><b style='color:#c0392b;'>❗ 重要注意事项：</b></p><ul><li>避免过度伸展</li><li>保持动作缓慢</li><li style='color:#c0392b;font-weight:800;'>如感到肌肉拉伤立即停止！</li></ul>"
                }
            }

            task_info = task_analysis.get(task_name, {
                "standard_practice": f"<p><b>{task_name}标准做法：</b></p><p>暂无解析信息</p>",
                "precautions": "<p><b style='color:#c0392b;'>❗ 重要注意事项：</b></p><p>暂无注意事项</p>"
            })

            combined_content = f"<div style='font-size:14pt; line-height:1.6;'>{task_info['standard_practice']}<div style='margin-top:20px;'>{task_info['precautions']}</div></div>"

            self.action_analysis_content.setTextFormat(Qt.RichText)
            self.action_analysis_content.setText(combined_content)

            # 更新样式
            self.action_analysis_content.setStyleSheet("""
                background-color: #fdf6e3;
                border-radius: 8px;
                padding: 15px;
                border: 2px solid #f1c40f;
            """)

    def toggle_detection(self):
        # 检查当前是否正在检测
        if not self.is_detecting:

            # 如果不在检测状态，则开始检测
            self.is_detecting = True
            # 将按钮文本改为“结束”，提示用户点击后将停止检测
            self.start_button.setText("🏃️ 结束训练")
            # 启动定时器，每隔30毫秒触发一次（用于定期执行某些任务）
            self.timer.start(30)
            # 加载示例视频以进行检测
            self.load_example_video()
            self.update_training_progress()  # 开始训练时初始化
            # self.update_action_analysis()
            # 重置计数器
            self.rehab_counter = RehabilitationCounter()  # 每次训练开始时创建新的计数器
        else:
            # 如果已经在检测状态，则停止检测
            self.is_detecting = False
            # 将按钮文本改为“开始”，提示用户点击后将启动检测
            self.start_button.setText("🏃️ 开始训练")
            # 停止定时器
            self.timer.stop()
            # 暂停媒体播放器中的视频
            self.media_player.pause()
            self.save_training_result()  # 新增保存调用
            QMessageBox.information(self, "训练结束", "训练数据已保存！")

    def update_frame(self):
        # 如果没有检测动作，则直接返回
        if not self.is_detecting:
            return
        # 从摄像头读取一帧
        ret, frame = self.cap.read()
        # 如果成功读取帧
        if ret:
            try:
                # 将原始帧和处理帧都调整为 350x250 大小
                original_frame = cv2.resize(frame, (350, 250))
                processed_frame = cv2.resize(frame, (350, 250))
                # 创建 OpenPose 的 Datum 对象
                datum = op.Datum()
                # 将处理后的帧设置为输入数据
                datum.cvInputData = processed_frame
                # 使用 OpenPose 处理帧
                self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                # 如果检测到人体关键点
                if datum.poseKeypoints.any():
                    # 获取关键点数据
                    keypoints = datum.poseKeypoints[0]
                    print("打印关键点数据")
                    print(keypoints)
                    # 设置当前任务配置
                    self.set_current_task_config()
                    print("未更新任务强度的任务配置")

                    # 获取当前任务的关键点 ID、目标角度范围和语音提示
                    joint_ids = self.current_task_config["joint_ids"]
                    print(joint_ids)
                    target_range = self.current_task_config["target_angle_range"]
                    print(target_range)
                    voice_prompts = self.current_task_config["voice_prompts"]

                    # 根据任务类型计算角度
                    if "neck" in joint_ids:
                        # 如果是脖颈屈伸任务
                        neck = keypoints[joint_ids["neck"]][:2]
                        shoulder = keypoints[joint_ids["shoulder"]][:2]
                        angle = calculate_neck_angle(neck, shoulder)
                    elif "hip" in joint_ids:
                        # 如果是大腿屈伸任务
                        hip = keypoints[joint_ids["hip"]][:2]
                        knee = keypoints[joint_ids["knee"]][:2]
                        ankle = keypoints[joint_ids["ankle"]][:2]
                        angle = calculate_leg_angle(hip, knee, ankle)
                    else:
                        # 默认为手臂屈伸任务
                        shoulder = keypoints[joint_ids["shoulder"]][:2]
                        elbow = keypoints[joint_ids["elbow"]][:2]
                        wrist = keypoints[joint_ids["wrist"]][:2]
                        angle = calculate_joint_angle(shoulder, elbow, wrist)

                    # 获取关键点的置信度
                    confidences = [keypoints[joint_id][2] for joint_id in joint_ids.values()]
                    # 如果所有关键点的置信度都高于阈值
                    if np.all(np.array(confidences) > REHAB_SETTINGS["confidence_threshold"]):
                        # 判断当前角度是否在目标范围内
                        status = target_range[0] <= angle <= target_range[1]
                        # 更新康复计数器
                        new_rep_completed = self.rehab_counter.update(status)

                        # 如果完成了一次新的重复动作
                        if new_rep_completed:
                            # 播放完成重复动作的语音提示
                            self.voice_assistant.speak_rep_complete(
                                voice_prompts["rep_complete"].format(self.rehab_counter.count)
                            )
                        # 如果没有在计数中
                        elif not self.rehab_counter.is_counting:
                            # 根据角度与目标范围的比较，播放相应的语音提示
                            if angle < target_range[0]:
                                self.voice_assistant.speak(voice_prompts["too_low"])

                            elif angle > target_range[1]:
                                self.voice_assistant.speak(voice_prompts["overbend"])

                            else:
                                self.voice_assistant.speak(voice_prompts["good_posture"])

                        # 计算得分，基于动作完成质量
                        if status:
                            score = 100.0
                            self.evaluation_content.setText(voice_prompts['good_posture'])
                        else:
                            if angle < target_range[0]:
                                score = max(0, 100.0 - (target_range[0] - angle) * 1.0)
                                self.evaluation_content.setText(voice_prompts["too_low"])
                            else:
                                score = max(0, 100.0 - (angle - target_range[1]) * 1.0)
                                self.evaluation_content.setText(voice_prompts["overbend"])

                        # 更新得分标签
                        self.score_label.setText(f"得分: {score:.4f}")
            except Exception as e:
                # 如果发生错误，打印错误信息
                print(f"更新帧时发生错误: {str(e)}")

        # 将原始帧从 BGR 转换为 RGB
        original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        # 获取图像的高、宽和通道数
        h, w, ch = original_rgb.shape
        # 计算每行的字节数
        bytes_per_line = ch * w
        # 将 OpenPose 处理后的帧从 BGR 转换为 RGB
        processed_rgb = cv2.cvtColor(datum.cvOutputData, cv2.COLOR_BGR2RGB)
        # 将处理后的帧转换为 QImage
        processed_qt_image = QImage(processed_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 将 QImage 转换为 QPixmap 并显示在 skeleton_label 上
        self.skeleton_label.setPixmap(QPixmap.fromImage(processed_qt_image))

    def save_training_result(self):
        # 连接到 SQLite 数据库 rehab.db
        conn = sqlite3.connect('rehab.db')
        # 创建一个游标对象，用于执行 SQL 查询
        cursor = conn.cursor()
        try:
            # 获取当前选中的任务描述
            selected_task = self.task_combo.currentText()
            # 如果有选中的任务
            if selected_task:
                # 提取任务名称
                task_name = selected_task.split(" (")[0]
                # 提取目标重复次数
                target_reps = int(selected_task.split("目标次数: ")[1].split(",")[0])

                # 查询当前任务的已完成重复次数和任务 ID
                cursor.execute("""
                    SELECT completed_reps, task_id
                    FROM RehabTasks
                    WHERE patient_id = ? AND task_description = ?
                """, (self.user_id, task_name))
                # 获取查询结果
                result = cursor.fetchone()
                print('zheshiwoderesult')
                print(result)
                # 如果查询到结果
                if result:
                    existing_reps, task_id = result
                else:
                    # 如果没有查询到结果，初始化已完成重复次数和任务 ID
                    existing_reps = 0
                    task_id = None

                # 计算总完成次数
                total_reps = existing_reps + self.rehab_counter.count

                # 如果总完成次数达到或超过目标次数
                if total_reps >= target_reps:
                    # 更新任务状态为 "Completed"
                    cursor.execute("""
                        UPDATE RehabTasks 
                        SET completed_reps = ?, status = ?
                        WHERE patient_id = ? AND task_description = ?
                    """, (total_reps, "Completed", self.user_id, task_name))
                else:
                    # 如果总完成次数大于 0 但未达到目标次数
                    if total_reps > 0:
                        # 更新任务状态为 "In Progress"
                        cursor.execute("""
                            UPDATE RehabTasks 
                            SET completed_reps = ?, status = ?
                            WHERE patient_id = ? AND task_description = ?
                        """, (total_reps, "In Progress", self.user_id, task_name))
                    else:
                        # 如果总完成次数为 0
                        cursor.execute("""
                            UPDATE RehabTasks 
                            SET completed_reps = ?
                            WHERE patient_id = ? AND task_description = ?
                        """, (total_reps, self.user_id, task_name))

                # 如果任务 ID 存在且完成次数大于 0
                if task_id and self.rehab_counter.count > 0:
                    # 获取当前得分
                    score = float(self.score_label.text().split(": ")[1])
                    # 修改 RehabTrainingUI 类中的 save_training_result 方法
                    cursor.execute("""
                        INSERT INTO TrainingRecords (task_id, score, reps_completed)
                        VALUES (?, ?, ?)
                    """, (task_id, score, self.rehab_counter.count))  # 添加 reps_completed 参数
                # 提交数据库更改
                conn.commit()

                # 重置康复计数器
                self.rehab_counter = RehabilitationCounter()
                self.update_training_progress()  # 保存后刷新进度
        except Exception as e:
            # 如果发生错误，打印错误信息
            print(f"保存训练结果时发生错误: {str(e)}")
        finally:
            # 关闭数据库连接
            conn.close()

    def closeEvent(self, event):
        try:
            self.is_detecting = False
            self.timer.stop()

            if self.cap.isOpened():
                self.cap.release()

            if hasattr(self, 'opWrapper'):
                self.opWrapper.stop()

            if hasattr(self, 'media_player'):
                self.media_player.stop()

            if hasattr(self, 'voice_assistant'):
                self.voice_assistant.engine.stop()

        except Exception as e:
            print(f"资源释放时发生错误: {str(e)}")

        event.accept()

    def show_training_history(self):
        dialog = TrainingHistoryDialog(self.user_id)
        dialog.exec_()


# 登录界面
class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RS - SIGN IN")
        self.setFixedSize(400, 600)  # 固定窗口尺寸

        # 主容器设置
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(40, 60, 40, 40)  # 边距调整
        main_layout.setSpacing(25)

        # ================= 标题区域 =================
        title_label = QLabel("SIGN IN")
        title_label.setStyleSheet("""
            font: bold 24px 'Arial';
            color: #2C3E50;
            qproperty-alignment: AlignCenter;
        """)
        main_layout.addWidget(title_label)

        # ================= 表单容器 =================
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(20)

        # 用户角色选择
        self.role_combo = QComboBox()
        self.role_combo.addItems(["康复训练师", "患者"])
        self.role_combo.setStyleSheet("""
            QComboBox {
                background: #FFFFFF;
                border: 2px solid #ECF0F1;
                padding: 12px 20px;
                font: bold 14px 'Arial';
                color: #34495E;

            }


            QComboBox::drop-down { width: 30px;}
        """)
        form_layout.addWidget(self.role_combo)

        # 用户名输入
        self.username_input = self._create_input_field("D:\文档\我的图片\用户1.png", "Username")
        form_layout.addWidget(self.username_input)

        # 密码输入
        self.password_input = self._create_password_field()
        form_layout.addWidget(self.password_input)

        # 记住密码 & 忘记密码
        self._create_remember_section(form_layout)

        # 登录按钮
        login_btn = QPushButton("SIGN IN")
        login_btn.setStyleSheet("""
            QPushButton {
                background: #2ECC71;
                color: white;
                border: none;
                border-radius: 15px;
                padding: 15px;
                font: bold 16px 'Arial';
            }
            QPushButton:hover { background: #27AE60; }
        """)
        login_btn.clicked.connect(self.login)
        form_layout.addWidget(login_btn)

        main_layout.addWidget(form_container)

        # ================= 社交登录区域 =================
        self._create_social_section(main_layout)

        # ================= 注册链接 =================
        self._create_register_link(main_layout)

    def _create_input_field(self, icon_path, placeholder):
        """创建带图标的输入框"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # 图标
        icon_label = QLabel()
        icon_label.setPixmap(QPixmap(icon_path).scaled(20, 20))
        layout.addWidget(icon_label)

        # 输入框
        input_field = QLineEdit()
        input_field.setPlaceholderText(placeholder)
        input_field.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                font: bold 14px 'Arial';
                color: #34495E;
                padding: 12px 0;
            }
        """)
        layout.addWidget(input_field, 1)

        # 底部装饰线
        line = QWidget()
        line.setFixedHeight(2)
        line.setStyleSheet("background: #ECF0F1;")
        layout.addWidget(line)

        return container

    def _create_password_field(self):
        """创建密码输入区域"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # 图标
        icon_label = QLabel()
        icon_label.setPixmap(QPixmap("D:\文档\我的图片\密码3.png").scaled(20, 20))
        layout.addWidget(icon_label)

        # 密码输入框
        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.Password)
        password_input.setPlaceholderText("Password")
        password_input.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                font: bold 14px 'Arial';
                color: #34495E;
                padding: 12px 0;

            }
        """)
        layout.addWidget(password_input, 1)

        # 显示密码切换
        show_check = QCheckBox("Show")
        show_check.setStyleSheet("""
            QCheckBox { color: #95A5A6; font: 12px 'Arial'; }
            QCheckBox::indicator { width: 16px; height: 16px; }
        """)
        show_check.stateChanged.connect(lambda s: password_input.setEchoMode(
            QLineEdit.Normal if s else QLineEdit.Password))
        layout.addWidget(show_check)

        # 底部装饰线
        line = QWidget()
        line.setFixedHeight(2)
        line.setStyleSheet("background: #ECF0F1;")
        layout.addWidget(line)

        return container

    def _create_remember_section(self, layout):
        """创建记住密码区域"""
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)

        # 记住我复选框
        remember_cb = QCheckBox("Stay logged in")
        remember_cb.setStyleSheet("""
            QCheckBox { color: #95A5A6; font: 12px 'Arial'; }
            QCheckBox::indicator { width: 16px; height: 16px; }
        """)
        hbox.addWidget(remember_cb)

        # # 忘记密码
        # forgot_btn = QPushButton("Reset Password")
        # forgot_btn.setStyleSheet("""
        #     QPushButton {
        #         color: #2ECC71;
        #         border: none;
        #         font: 12px 'Arial';
        #         padding: 0;
        #     }
        #     QPushButton:hover { color: #27AE60; }
        # """)
        # hbox.addWidget(forgot_btn)
        #
        # layout.addWidget(container)

    def _create_social_section(self, layout):
        """创建社交登录区域"""
        # 分隔文字
        divider = QLabel("or sign in with")
        divider.setStyleSheet("""
            color: #BDC3C7;
            font: 14px 'Arial';
            qproperty-alignment: AlignCenter;
        """)
        layout.addWidget(divider)

        # 社交按钮容器
        btn_container = QWidget()
        hbox = QHBoxLayout(btn_container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(15)

        # 社交按钮
        social_btns = [
            ("D:\文档\我的图片\微信.png", "#F8F8FF"),
            ("D:\文档\我的图片\qq.png", "#F8F8FF"),
            ("D:\文档\我的图片\微博.png", "#F8F8FF")
        ]

        for icon, color in social_btns:
            btn = QPushButton()
            btn.setFixedSize(50, 50)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {color};
                    border-radius: 25px;
                }}
                QPushButton:hover {{ background: {color}DD; }}
            """)
            btn.setIcon(QIcon(icon))
            btn.setIconSize(btn.size())
            hbox.addWidget(btn)

        layout.addWidget(btn_container)

    def _create_register_link(self, layout):
        """创建注册链接"""
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)

        prompt = QLabel("Don't have an account?")
        prompt.setStyleSheet("color: #95A5A6; font: 14px 'Arial';")

        reg_btn = QPushButton("Sign up")
        reg_btn.setStyleSheet("""
            QPushButton {
                color: #2ECC71;
                border: none;
                font: bold 14px 'Arial';
                padding: 0;
            }
            QPushButton:hover { color: #27AE60; }
        """)
        reg_btn.clicked.connect(self.show_register)

        hbox.addWidget(prompt)
        hbox.addWidget(reg_btn)
        layout.addWidget(container)

    def login(self):
        # 保留原有登录逻辑
        role = self.role_combo.currentText()
        username = self.username_input.findChild(QLineEdit).text()
        password = self.password_input.findChild(QLineEdit).text()

        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            if user[3] == role:
                if role == "康复训练师":
                    self.therapist_window = TherapistWindow(self)
                    self.therapist_window.show()
                    self.close()
                else:
                    self.patient_window = RehabTrainingUI(self, user[0])
                    self.patient_window.show()
                    self.close()
            else:
                QMessageBox.warning(self, "登录失败", "用户角色不匹配，请重试！")
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误，请重试！")

    def show_register(self):
        self.register_window = RegisterWindow()
        self.register_window.exec_()

    def show_register(self):
        self.register_ui = RegisterWindow()
        self.register_ui.exec_()


class RegisterWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RS - Register")
        self.setFixedSize(533, 800)
        self.setStyleSheet("QDialog { background: #ffffff; }")

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 30, 30)
        main_layout.setSpacing(15)

        # ======== 标题栏 ========
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel("康复训练系统 - 注册")
        title_label.setStyleSheet("font: bold 18px; color: #333333;")
        title_layout.addWidget(title_label)
        main_layout.addWidget(title_widget)

        # ======== 滚动区域 ========
        scroll_area = QScrollArea()  # 创建滚动区域
        scroll_area.setWidgetResizable(True)  # 设置滚动区域的内容可调整大小
        scroll_content = QWidget()  # 创建一个QWidget用于放置滚动区域的内容
        scroll_layout = QVBoxLayout(scroll_content)  # 创建垂直布局用于放置滚动区域的内容
        scroll_layout.setSpacing(20)  # 设置滚动区域布局中各部件之间的间距

        # 用户图标（添加本地图标）
        icon_button = QPushButton()
        icon_button.setIcon(QIcon("D:\文档\我的图片\注册5.png"))  # 使用本机图标路径
        icon_button.setIconSize(QSize(80, 80))
        icon_button.setStyleSheet(
            "QPushButton { background: #f5f5f5; border-radius: 40px; border: 2px dashed #cccccc; }")
        scroll_layout.addWidget(icon_button, 0, Qt.AlignCenter)

        # ======== 表单区域 ========
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setContentsMargins(40, 20, 40, 20)
        form_layout.setSpacing(15)

        # 标题文字
        welcome_label = QLabel("Welcome!")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font: bold 28px; color: #333333;")
        form_layout.addWidget(welcome_label)

        subtitle_label = QLabel("You're going to sign up to rehabilitation!")
        subtitle_label.setWordWrap(True)
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font: bold 20px; color: #666666;")
        form_layout.addWidget(subtitle_label)

        # 输入字段样式
        input_style = "QLineEdit { background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; }"
        combo_style = "QComboBox { background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; }"

        # 创建带图标、标签和输入框的辅助函数
        def create_input_row(icon_path, label_text, placeholder_text):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setSpacing(10)
            layout.setContentsMargins(0, 0, 0, 0)

            # 图标
            icon_label = QLabel()
            icon_label.setPixmap(QPixmap(icon_path).scaled(24, 24, Qt.KeepAspectRatio))
            layout.addWidget(icon_label)

            # 标签
            label = QLabel(label_text)
            label.setStyleSheet("QLabel { font: bold 14px; color: #333333; }")
            layout.addWidget(label)

            # 输入框
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(placeholder_text)
            line_edit.setStyleSheet(input_style)
            layout.addWidget(line_edit, 1)  # 设置伸缩因子为1，占据剩余空间

            return widget, line_edit

        # 创建带图标、标签和下拉框的辅助函数
        def create_combo_row(icon_path, label_text, items):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setSpacing(10)
            layout.setContentsMargins(0, 0, 0, 0)

            # 图标
            icon_label = QLabel()
            icon_label.setPixmap(QPixmap(icon_path).scaled(24, 24, Qt.KeepAspectRatio))
            layout.addWidget(icon_label)

            # 标签
            label = QLabel(label_text)
            label.setStyleSheet("QLabel { font: bold 14px; color: #333333; }")
            layout.addWidget(label)

            # 下拉框
            combo = QComboBox()
            combo.addItems(items)
            combo.setStyleSheet(combo_style)
            layout.addWidget(combo, 1)  # 设置伸缩因子为1，占据剩余空间

            return widget, combo

        # ======== 用户名 ========
        username_widget, self.username_edit = create_input_row(
            "D:\文档\我的图片\用户名.png", "用户:", "用户名"
        )
        form_layout.addWidget(username_widget)

        # ======== 密码 ========
        password_widget, self.password_edit = create_input_row(
            "D:\文档\我的图片\密码.png", "密码:", "密码"
        )
        self.password_edit.setEchoMode(QLineEdit.Password)
        form_layout.addWidget(password_widget)

        # ======== 确认密码 ========
        confirm_widget, self.confirm_edit = create_input_row(
            "D:\文档\我的图片\确认密码.png", "确认:", "确认密码"
        )
        self.confirm_edit.setEchoMode(QLineEdit.Password)
        form_layout.addWidget(confirm_widget)

        # ======== 角色选择区域 ========
        role_widget, self.role_combo = create_combo_row(
            "D:\文档\我的图片\角色选择.png", "角色:", ["请选择角色", "康复训练师", "患者"]
        )
        form_layout.addWidget(role_widget)

        # ======== 患者详细信息区域 ========
        self.patient_details = QWidget()
        patient_layout = QVBoxLayout(self.patient_details)
        patient_layout.setContentsMargins(0, 20, 0, 0)

        # ======== 年龄 ========
        age_widget, self.age_edit = create_input_row(
            "D:\文档\我的图片\年龄.png", "年龄:", "请输入年龄"
        )
        patient_layout.addWidget(age_widget)

        # ======== 性别 ========
        gender_widget = QWidget()
        gender_layout = QHBoxLayout(gender_widget)
        gender_layout.setSpacing(10)
        gender_layout.setContentsMargins(0, 0, 0, 0)

        # 图标
        gender_icon = QLabel()
        gender_icon.setPixmap(QPixmap("D:\文档\我的图片\性别.png").scaled(24, 24, Qt.KeepAspectRatio))
        gender_layout.addWidget(gender_icon)

        # 标签
        gender_label = QLabel("性别:")
        gender_label.setStyleSheet("QLabel { font: bold 14px; color: #333333; }")
        gender_layout.addWidget(gender_label)

        # 下拉框
        self.gender_edit = QComboBox()
        self.gender_edit.addItems(["男", "女", "其他"])
        self.gender_edit.setStyleSheet(combo_style)
        gender_layout.addWidget(self.gender_edit, 1)
        patient_layout.addWidget(gender_widget)

        # ======== 身高 ========
        height_widget, self.height_edit = create_input_row(
            "D:\文档\我的图片\身高.png", "身高:", "身高(cm)"
        )
        patient_layout.addWidget(height_widget)

        # ======== 体重 ========
        weight_widget, self.weight_edit = create_input_row(
            "D:\文档\我的图片\体重.png", "体重:", "体重"
        )
        patient_layout.addWidget(weight_widget)

        # ======== 家庭地址 ========
        address_label = QLabel("家庭地址:")
        address_label.setStyleSheet("QLabel { font: bold 14px; color: #333333; }")
        patient_layout.addWidget(address_label)

        # 省份下拉框
        province_widget, self.province_combo = create_combo_row(
            "D:\文档\我的图片\省份.png", "省份:", ["北京", "上海", "广东", "江苏", "浙江", "河南", "河北", "其他"]
        )
        patient_layout.addWidget(province_widget)

        # 城市下拉框
        city_widget, self.city_combo = create_combo_row(
            "D:\文档\我的图片\城市.png", "城市:", []
        )
        patient_layout.addWidget(city_widget)

        # 区县下拉框
        district_widget, self.district_combo = create_combo_row(
            "D:\文档\我的图片\区县.png", "区县:", []
        )
        patient_layout.addWidget(district_widget)

        # 详细地址
        detail_address_widget, self.detail_address_edit = create_input_row(
            "D:\文档\我的图片\地址.png", "地址:", "街道、门牌号等"
        )
        patient_layout.addWidget(detail_address_widget)

        # ======== 电话 ========
        phone_widget, self.phone_edit = create_input_row(
            "D:\文档\我的图片\电话.png", "电话:", "电话"
        )
        patient_layout.addWidget(phone_widget)

        # ======== 紧急联系人 ========
        emergency_contact_widget, self.emergency_contact_edit = create_input_row(
            "D:\文档\我的图片\紧急联系人.png", "紧急:", "紧急联系人"
        )
        patient_layout.addWidget(emergency_contact_widget)

        # ======== 紧急联系人电话 ========
        emergency_phone_widget, self.emergency_phone_edit = create_input_row(
            "D:\文档\我的图片\紧急联系人电话.png", "电话:", "紧急联系人电话"
        )
        patient_layout.addWidget(emergency_phone_widget)

        # ======== 静息心率 ========
        resting_hr_widget, self.resting_hr_edit = create_input_row(
            "D:\文档\我的图片\静息心率.png", "心率:", "静息心率"
        )
        patient_layout.addWidget(resting_hr_widget)

        # ======== 血压 ========
        blood_pressure_widget, self.blood_pressure_edit = create_input_row(
            "D:\文档\我的图片\血压.png", "血压:", "血压 (mmHg)"
        )
        patient_layout.addWidget(blood_pressure_widget)

        # ======== 血氧饱和度 ========
        blood_oxygen_widget, self.blood_oxygen_edit = create_input_row(
            "D:\文档\我的图片\血氧饱和度.png", "血氧:", "血氧饱和度 (%)"
        )
        patient_layout.addWidget(blood_oxygen_widget)

        # ======== 病种分类 ========
        condition_widget = QWidget()
        condition_layout = QHBoxLayout(condition_widget)
        condition_layout.setSpacing(10)
        condition_layout.setContentsMargins(0, 0, 0, 0)

        # 图标
        condition_icon = QLabel()
        condition_icon.setPixmap(QPixmap("D:\文档\我的图片\病种分类.png").scaled(24, 24, Qt.KeepAspectRatio))
        condition_layout.addWidget(condition_icon)

        # 标签
        condition_label = QLabel("分类:")
        condition_label.setStyleSheet("QLabel { font: bold 14px; color: #333333; }")
        condition_layout.addWidget(condition_label)

        # 下拉框
        self.condition_edit = QComboBox()
        self.condition_edit.addItems(["脑卒中偏瘫", "骨关节疾病", "发育障碍"])
        self.condition_edit.setStyleSheet(combo_style)
        condition_layout.addWidget(self.condition_edit, 1)
        patient_layout.addWidget(condition_widget)

        form_layout.addWidget(self.patient_details)

        # 注册按钮
        register_btn = QPushButton("注册")
        register_btn.setStyleSheet(
            "QPushButton { background: #4CAF50; color: white; border: none; border-radius: 15px; padding: 15px; font: bold 20px; }")
        register_btn.setCursor(Qt.PointingHandCursor)
        register_btn.clicked.connect(self.register)
        form_layout.addWidget(register_btn)

        scroll_layout.addWidget(form_widget)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # 默认隐藏患者详细信息区域
        self.patient_details.setVisible(False)

        # 连接角色选择事件
        self.role_combo.currentIndexChanged.connect(self.role_changed)

        # 连接省份选择事件
        self.province_combo.currentIndexChanged.connect(self.update_cities)

        # 连接城市选择事件
        self.city_combo.currentIndexChanged.connect(self.update_districts)

    def role_changed(self):
        role = self.role_combo.currentText()
        if role == "患者":
            self.patient_details.setVisible(True)
        else:
            self.patient_details.setVisible(False)

    def update_cities(self):
        # 根据选择的省份更新城市下拉框
        selected_province = self.province_combo.currentText()
        # 示例数据，实际应用中可以根据需求扩展
        city_data = {
            "北京": ["北京"],
            "上海": ["上海"],
            "广东": ["广州", "深圳", "佛山", "东莞", "珠海", "中山", "江门", "惠州", "其他"],
            "江苏": ["南京", "苏州", "无锡", "常州", "徐州", "连云港", "淮安", "盐城", "扬州", "镇江", "泰州", "宿迁", "南通", "其他"],
            "浙江": ["杭州", "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "衢州", "舟山", "台州", "丽水", "其他"],
            "河南": ["郑州", "开封", "洛阳", "平顶山", "安阳", "鹤壁", "新乡", "焦作", "濮阳", "许昌", "漯河", "三门峡", "南阳", "商丘", "信阳", "周口",
                   "驻马店", "济源", "其他"],
            "河北": ["石家庄", "唐山", "邯郸", "秦皇岛", "保定", "张家口", "承德", "廊坊", "沧州", "衡水", "邢台", "其他"],
            "其他": ["其他"]
        }
        self.city_combo.clear()
        self.city_combo.addItems(city_data.get(selected_province, ["其他"]))

    def update_districts(self):
        # 根据选择的城市更新区县下拉框
        selected_city = self.city_combo.currentText()
        # 示例数据，实际应用中可以根据需求扩展
        district_data = {
            # 河南省区县数据
            "郑州": ["中原区", "二七区", "管城回族区", "金水区", "上街区", "惠济区", "中牟县", "荥阳市", "新密市", "新郑市", "登封市", "巩义市", "其他"],
            "开封": ["龙亭区", "顺河回族区", "鼓楼区", "禹王台区", "祥符区", "杞县", "通许县", "尉氏县", "兰考县", "其他"],
            "洛阳": ["老城区", "西工区", "瀍河回族区", "涧西区", "洛龙区", "偃师区", "孟津区", "新安县", "栾川县", "嵩县", "汝阳县", "宜阳县", "洛宁县", "伊川县",
                   "其他"],
            "平顶山": ["新华区", "卫东区", "石龙区", "湛河区", "宝丰县", "叶县", "鲁山县", "郏县", "舞钢市", "汝州市", "其他"],
            "安阳": ["文峰区", "北关区", "殷都区", "龙安区", "安阳县", "汤阴县", "内黄县", "滑县", "林州市", "其他"],
            "鹤壁": ["鹤山区", "山城区", "淇滨区", "浚县", "淇县", "其他"],
            "新乡": ["红旗区", "卫滨区", "凤泉区", "牧野区", "新乡县", "获嘉县", "原阳县", "延津县", "封丘县", "卫辉市", "辉县市", "长垣市", "其他"],
            "焦作": ["解放区", "中站区", "马村区", "山阳区", "修武县", "博爱县", "武陟县", "温县", "沁阳市", "孟州市", "其他"],
            "濮阳": ["华龙区", "清丰县", "南乐县", "范县", "台前县", "濮阳县", "其他"],
            "许昌": ["魏都区", "建安区", "鄢陵县", "襄城县", "禹州市", "长葛市", "其他"],
            "漯河": ["源汇区", "郾城区", "召陵区", "舞阳县", "临颍县", "其他"],
            "三门峡": ["湖滨区", "陕州区", "渑池县", "卢氏县", "义马市", "灵宝市", "其他"],
            "南阳": ["宛城区", "卧龙区", "南召县", "方城县", "西峡县", "镇平县", "内乡县", "淅川县", "社旗县", "唐河县", "新野县", "桐柏县", "邓州市", "其他"],
            "商丘": ["梁园区", "睢阳区", "民权县", "睢县", "宁陵县", "柘城县", "虞城县", "夏邑县", "永城市", "其他"],
            "信阳": ["浉河区", "平桥区", "罗山县", "光山县", "新县", "商城县", "固始县", "潢川县", "淮滨县", "息县", "其他"],
            "周口": ["川汇区", "扶沟县", "西华县", "商水县", "沈丘县", "郸城县", "鹿邑县", "太康县", "淮阳区", "项城市", "其他"],
            "驻马店": ["驿城区", "上蔡县", "西平县", "遂平县", "正阳县", "确山县", "泌阳县", "汝南县", "新蔡县", "其他"],
            "济源": ["济源市", "其他"],

            # 河北省区县数据
            "石家庄": ["长安区", "桥西区", "新华区", "裕华区", "鹿泉区", "栾城区", "藁城区", "井陉矿区", "正定县", "行唐县", "灵寿县", "高邑县", "深泽县", "赞皇县",
                    "无极县", "平山县", "元氏县", "赵县", "辛集市", "藁城市", "晋州市", "新乐市", "其他"],
            "唐山": ["路南区", "路北区", "古冶区", "开平区", "丰南区", "丰润区", "曹妃甸区", "遵化市", "迁安市", "滦州市", "滦南县", "乐亭县", "迁西县", "玉田县",
                   "其他"],
            "邯郸": ["邯山区", "丛台区", "复兴区", "峰峰矿区", "肥乡区", "永年区", "武安市", "涉县", "磁县", "邱县", "鸡泽县", "广平县", "成安县", "曲周县",
                   "馆陶县", "魏县", "大名县", "其他"],
            "秦皇岛": ["海港区", "山海关区", "北戴河区", "抚宁区", "昌黎县", "卢龙县", "青龙满族自治县", "其他"],
            "保定": ["竞秀区", "莲池区", "满城区", "清苑区", "徐水区", "涿州市", "定州市", "安国市", "高碑店市", "曲阳县", "涞水县", "阜平县", "顺平县", "唐县",
                   "望都县", "高阳县", "定兴县", "涞源县", "易县", "曲阳县", "蠡县", "博野县", "雄县", "其他"],
            "张家口": ["桥东区", "桥西区", "宣化区", "下花园区", "万全区", "崇礼区", "张北县", "康保县", "沽源县", "尚义县", "蔚县", "阳原县", "怀安县", "怀来县",
                    "涿鹿县", "其他"],
            "承德": ["双桥区", "双滦区", "鹰手营子矿区", "承德县", "兴隆县", "滦平县", "隆化县", "平泉市", "宽城满族自治县", "围场满族蒙古族自治县", "其他"],
            "廊坊": ["安次区", "广阳区", "霸州市", "三河市", "固安县", "永清县", "香河县", "大厂回族自治县", "文安县", "大城县", "其他"],
            "沧州": ["新华区", "运河区", "泊头市", "任丘市", "黄骅市", "河间市", "沧县", "青县", "东光县", "海兴县", "盐山县", "肃宁县", "献县", "孟村回族自治县",
                   "其他"],
            "衡水": ["桃城区", "冀州区", "深州市", "枣强县", "武邑县", "武强县", "饶阳县", "安平县", "故城县", "景县", "阜城县", "其他"],
            "邢台": ["襄都区", "信都区", "南和区", "任泽区", "沙河市", "临西县", "内丘县", "柏乡县", "隆尧县", "巨鹿县", "新河县", "广宗县", "平乡县", "威县",
                   "清河县", "南宫市", "其他"],

            # 其他省份的区县数据（示例）
            "北京": ["东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区", "门头沟区", "房山区", "通州区", "顺义区", "昌平区", "大兴区", "怀柔区", "平谷区",
                   "密云区", "延庆区", "其他"],
            "上海": ["黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区", "杨浦区", "闵行区", "宝山区", "嘉定区", "浦东新区", "金山区", "松江区", "青浦区",
                   "奉贤区", "崇明区", "其他"],
            "广州": ["越秀区", "荔湾区", "海珠区", "天河区", "白云区", "黄埔区", "番禺区", "花都区", "南沙区", "从化区", "增城区", "其他"],
            "深圳": ["罗湖区", "福田区", "南山区", "盐田区", "宝安区", "龙岗区", "龙华区", "坪山区", "光明区", "大鹏新区", "其他"],
            "其他": ["其他"]
        }
        self.district_combo.clear()
        self.district_combo.addItems(district_data.get(selected_city, ["其他"]))

    def register(self):
        username = self.username_edit.text()
        password = self.password_edit.text()
        confirm_password = self.confirm_edit.text()

        if not username or not password or not confirm_password:
            QMessageBox.warning(self, "注册失败", "请输入完整的用户信息！")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "注册失败", "两次输入的密码不一致！")
            return

        # 获取选择的角色
        role = self.role_combo.currentText()
        print("1")
        print(role)
        if role == "请选择角色":
            QMessageBox.warning(self, "注册失败", "请选择用户角色！")
            return

        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()

        try:
            # 插入用户基本信息
            cursor.execute("INSERT INTO Users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
            user_id = cursor.lastrowid

            # 如果是患者，插入详细信息
            if role == "患者":
                age = self.age_edit.text()
                gender = self.gender_edit.currentText()
                height = self.height_edit.text()
                weight = self.weight_edit.text()
                province = self.province_combo.currentText()
                city = self.city_combo.currentText()
                district = self.district_combo.currentText()
                detail_address = self.detail_address_edit.text()
                address = f"{province}, {city}, {district}, {detail_address}"
                phone = self.phone_edit.text()
                emergency_contact = self.emergency_contact_edit.text()
                emergency_phone = self.emergency_phone_edit.text()
                resting_hr = self.resting_hr_edit.text()
                blood_pressure = self.blood_pressure_edit.text()
                blood_oxygen = self.blood_oxygen_edit.text()
                condition = self.condition_edit.currentText()

                cursor.execute(
                    "INSERT INTO PatientProfiles (user_id, age, gender, height, weight, address, phone, emergency_contact, emergency_phone, resting_hr, blood_pressure, blood_oxygen, condition) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (user_id, age, gender, height, weight, address, phone, emergency_contact, emergency_phone,
                     resting_hr, blood_pressure, blood_oxygen, condition)
                )

            conn.commit()
            QMessageBox.information(self, "注册成功", "注册成功！您可以使用新账号登录。")
            self.accept()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "注册失败", "用户名已存在，请换一个用户名！")
        finally:
            conn.close()


# 初始化数据库
def init_database():
    conn = sqlite3.connect('rehab.db')
    cursor = conn.cursor()

    # cursor.execute("DROP TABLE IF EXISTS Users")
    # cursor.execute("DROP TABLE IF EXISTS RehabTasks")
    # cursor.execute("DROP TABLE IF EXISTS TrainingRecords")

    # 创建 Users 表，添加 role 字段
    cursor.execute('''CREATE TABLE IF NOT EXISTS Users (
                      user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      role TEXT NOT NULL)''')

    # 创建 PatientProfiles 表，存储患者详细信息
    cursor.execute('''CREATE TABLE IF NOT EXISTS PatientProfiles (
                      profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      age INTEGER,
                      gender TEXT,
                      height REAL,
                      weight REAL,
                      address TEXT,
                      phone TEXT,
                      emergency_contact TEXT,
                      emergency_phone TEXT,
                      resting_hr INTEGER,
                      blood_pressure TEXT,
                      blood_oxygen REAL,
                      condition TEXT,
                      FOREIGN KEY (user_id) REFERENCES Users(user_id))''')

    # 创建 RehabTasks 和 TrainingRecords 表（保持原有结构）
    cursor.execute('''CREATE TABLE IF NOT EXISTS RehabTasks (
                      task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                      therapist_id INTEGER NOT NULL,
                      patient_id INTEGER NOT NULL,
                      task_description TEXT,
                      target_reps INTEGER,
                      completed_reps INTEGER DEFAULT 0,
                      due_date DATE,
                      status TEXT DEFAULT 'Not Started',
                      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (therapist_id) REFERENCES Users(user_id),
                      FOREIGN KEY (patient_id) REFERENCES Users(user_id))''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS TrainingRecords (
                      record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                      task_id INTEGER NOT NULL,
                      score REAL,
                      reps_completed INTEGER DEFAULT 0,
                      time DATETIME DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (task_id) REFERENCES RehabTasks(task_id))''')

    conn.commit()
    conn.close()


# 训练历史对话框类
class TrainingHistoryDialog(QDialog):
    def __init__(self, user_id):
        super().__init__()
        # 基础设置
        self.setWindowTitle("训练历史")
        self.setGeometry(370, 35, 1240, 970)
        self.user_id = user_id

        # 全局字体设置
        font = QFont("Segoe UI", 10)
        self.setFont(font)

        # 现代化配色方案
        self.setStyleSheet("""
                   QDialog {
                       background: #f8f9fa;
                   }
                   QGroupBox {
                       border: 2px solid #e0e0e0;
                       border-radius: 8px;
                       margin-top: 20px;
                       padding: 10px;
                   }
                   QGroupBox::title {
                       subcontrol-origin: margin;
                       left: 10px;
                       padding: 0 5px;
                       color: #495057;
                       font: bold 14px 'Segoe UI';
                   }
                   QTableWidget {
                       background: white;
                       border: 1px solid #dee2e6;
                       border-radius: 6px;
                       gridline-color: #e9ecef;
                       selection-background-color: #e3f2fd;
                   }
                   QHeaderView::section {
                       background: #f1f3f5;
                       padding: 12px;
                       border: none;
                       font: 500 12px 'Segoe UI';
                       color: #495057;
                   }
                   QTableWidget::item {
                       padding: 10px;
                       border-bottom: 1px solid #e9ecef;
                   }
                   QPushButton {
                       background: #4dabf7;
                       color: white;
                       border-radius: 6px;
                       padding: 10px 20px;
                       font: 500 12px 'Segoe UI';
                       min-width: 80px;
                   }
                   QPushButton:hover {
                       background: #339af0;
                   }
                   QPushButton:pressed {
                       background: #228be6;
                   }
               """)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 25, 25, 25)
        # main_layout.setSpacing(20)

        # 标题区域
        header = QLabel("📊 训练历史")
        header.setStyleSheet("""
                    QLabel {
                        font: bold 28px 'Segoe UI';
                        color: #343a40;
                        qproperty-alignment: AlignCenter;
                        padding: 15px 0;
                    }
                """)
        main_layout.addWidget(header)



        # 操作按钮区
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        self.export_button = QPushButton("导出数据")
        self.export_button.setIcon(QIcon(":/icons/export.svg"))  # 建议使用SVG图标
        self.export_button.setIconSize(QSize(20, 20))
        self.export_button.setCursor(Qt.PointingHandCursor)
        btn_layout.addStretch()
        btn_layout.addWidget(self.export_button)

        main_layout.addWidget(btn_container)

        # 数据表格
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["训练项目", "完成次数", "评估得分", "训练时间"])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setHighlightSections(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setStyleSheet("""
                    QTableWidget {
                        alternate-background-color: #f8f9fa;
                    }
                """)

        # 图表设置
        self.figure = Figure(facecolor='none', figsize=(12, 5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background: transparent;")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#f8f9fa')

        # # 新增AI训练建议区域
        # self.ai_suggestion_group = QGroupBox("AI训练建议")
        # self.ai_suggestion_group.setStyleSheet("""
        #             QGroupBox {
        #                 background: #f8f9fa;
        #                 border: 1px solid #e0e0e0;
        #                 border-radius: 8px;
        #                 margin-top: 20px;
        #                 padding: 10px;
        #             }
        #         """)
        #
        # suggestion_layout = QVBoxLayout(self.ai_suggestion_group)
        #
        # self.suggestion_label = QLabel("加载中，请稍候...")
        # self.suggestion_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # self.suggestion_label.setWordWrap(True)
        # self.suggestion_label.setStyleSheet("font: 14px 'Segoe UI'; color: #333333; padding: 10px;")
        #
        # suggestion_layout.addWidget(self.suggestion_label)
        #
        # main_layout.addWidget(self.ai_suggestion_group)
        # main_layout.setStretch(6, 20)  # 建议区域占比30%

        # 修改后的AI建议区域代码
        self.ai_suggestion_group = QGroupBox("AI训练建议")
        self.ai_suggestion_group.setStyleSheet("""
            QGroupBox {
                background: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 20px;
                padding: 10px;
            }
        """)

        # 创建滚动区域容器
        scroll_container = QScrollArea()
        scroll_container.setWidgetResizable(True)
        scroll_container.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_container.setStyleSheet("background: transparent; border: none;")

        # 在样式表中添加滚动条美化
        scroll_container.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                width: 10px;
                background: #f1f3f5;
            }
            QScrollBar::handle:vertical {
                background: #ced4da;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # 创建内容容器
        suggestion_content = QWidget()
        suggestion_layout = QVBoxLayout(suggestion_content)
        suggestion_layout.setContentsMargins(5, 5, 5, 5)

        self.suggestion_label = QLabel("加载中，请稍候...")
        self.suggestion_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 设置样式
        self.suggestion_label.setStyleSheet("""
            QLabel {
                font: 14px 'Segoe UI';
                color: #333333;
                padding: 10px;
                background: white;
                border-radius: 6px;
                margin: 5px;
            }
        """)

        # 组装滚动区域
        suggestion_layout.addWidget(self.suggestion_label)
        scroll_container.setWidget(suggestion_content)
        self.ai_suggestion_group.setLayout(QVBoxLayout())
        self.ai_suggestion_group.layout().addWidget(scroll_container)

        # 在main_layout中添加
        main_layout.addWidget(self.ai_suggestion_group)
        # main_layout.setStretchFactor(self.ai_suggestion_group, 40)  # 调整布局占比

        # 加载AI训练建议
        self.load_ai_suggestions()

        # # 图表设置
        # self.figure = Figure(facecolor='none', figsize=(12, 5))
        # self.canvas = FigureCanvas(self.figure)
        # self.canvas.setStyleSheet("background: transparent;")
        # self.ax = self.figure.add_subplot(111)
        # self.ax.set_facecolor('#f8f9fa')

        # 创建水平布局容器
        chart_table_container = QWidget()
        h_layout = QHBoxLayout(chart_table_container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(40)

        # 布局组织
        table_group = QGroupBox("训练记录")
        table_group.setLayout(QVBoxLayout())
        table_group.layout().addWidget(self.history_table)

        chart_group = QGroupBox("得分趋势图")
        chart_group.setLayout(QVBoxLayout())
        chart_group.layout().addWidget(self.canvas)

        # 新水平布局
        h_layout.addWidget(table_group, 50)  # 表格占45%
        h_layout.addWidget(chart_group, 50)  # 图表占55%
        main_layout.addWidget(chart_table_container,60)  # 整体容器

        # main_layout.addWidget(table_group)
        # main_layout.addWidget(chart_group)
        # main_layout.setStretch(3, 50)  # 表格区域占比40%
        # main_layout.setStretch(4, 50)  # 图表区域占比60%
        # 修改布局比例设置（原main_layout.setStretch(6, 20)）
        # main_layout.setStretchFactor(table_group, 35)  # 表格区域占比35%
        # main_layout.setStretchFactor(chart_group, 45)  # 图表区域占比45%
        main_layout.setStretchFactor(self.ai_suggestion_group, 40)  # 建议区域占比20%

        # 信号连接
        self.export_button.clicked.connect(self.on_export_clicked)
        # 加载数据
        self.load_training_history()






    def on_export_clicked(self):
        filename, _ = QFileDialog.getSaveFileName(self, "导出文件", "", "CSV Files (*.csv);;All Files (*)")
        if filename:
            try:
                self.export_to_csv(filename)
                QMessageBox.information(self, "导出成功", f"数据已成功导出到 {filename}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出过程中发生错误: {str(e)}")

    def export_to_csv(self, filename):
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header_labels = []
            for column in range(self.history_table.columnCount()):
                header_item = self.history_table.horizontalHeaderItem(column)
                if header_item:
                    header_labels.append(header_item.text())
                else:
                    header_labels.append(f"Column {column}")
            writer.writerow(header_labels)
            for row in range(self.history_table.rowCount()):
                row_data = []
                for column in range(self.history_table.columnCount()):
                    item = self.history_table.item(row, column)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
                writer.writerow(row_data)

    def load_ai_suggestions(self):
        # 获取用户信息
        user_profile = self.load_user_profile()
        training_history = self.load_training_history_forAI()
        print("已经获取用户信息，准备调用API")
        print("打印获取的用户信息和训练历史——————————————————————————————————————————————————————————————————")
        print(user_profile)
        print(training_history)
        print("————————————————————————————————————————结束打印————————————————————————————————————————————")
        if user_profile and training_history:
            try:
                # 调用大模型API，这里采用异步非阻塞方式
                threading.Thread(target=self.fetch_ai_suggestions, args=(user_profile, training_history))
                thread = threading.Thread(target=self.fetch_ai_suggestions, args=(user_profile, training_history))
                thread.start()
                # thread.daemon = True
                print("线程创建成功")  # 如果能看到这个打印，说明线程创建成功
                thread.start()
                print("线程启动")  # 如果能看到这个打印，说明线程启动指令已发出
            except Exception as e:
                print(f"AI建议加载失败: {e}")

    def fetch_ai_suggestions(self, user_profile, training_history):
        print("镇北调用个人信息")
        try:
            # 准备用户信息和训练历史数据
            user_info = {
                "age": user_profile.get("age", "未知"),
                "gender": user_profile.get("gender", "未知"),
                "height": user_profile.get("height", "未知"),
                "weight": user_profile.get("weight", "未知"),
                "condition": user_profile.get("condition", "无特殊病史")
            }

            print("打印API-INPUT")
            print(user_info)
            print("打印API-INPUT结束")

            # 准备API请求内容
            api_input = f"""
                用户基本信息：
                - 年龄：{user_info["age"]}
                - 性别：{user_info["gender"]}
                - 身高：{user_info["height"]} cm
                - 体重：{user_info["weight"]} kg
                - 基础病史：{user_info["condition"]}

                训练历史：
                {self.format_training_history(training_history)}

                请根据以上信息，为用户制定个性化的康复训练建议。建议内容应包括：
                1. 当前训练效果评估
                2. 训练强度调整建议
                3. 训练频率建议
                4. 需要改进的方面
                5. 下一步训练计划
            """
            print("打印API-INPUT")
            print(api_input)
            print("打印API-INPUT结束")

            # 调用大模型API
            api_response = self.call_bailian_api(api_input)
            print("已经调用大模型，以下是输入信息")
            print(api_input)
            print("输入信息打印完成")

            # 更新UI（需要切换到主线程）
            self.update_suggestion_label(api_response)

        except Exception as e:
            print(f"AI建议获取失败: {e}")
            self.update_suggestion_label(f"获取AI建议失败：{str(e)}")

    def call_bailian_api(self, input_text):
        """调用百炼大模型API"""
        api_key = 'sk-39892aafc154458b90c494a37eb7f86e'  # 配置您的API Key
        try:
            response = dashscope.Generation.call(
                api_key=api_key,
                model="qwen-max",
                messages=[
                    {"role": "system", "content": "你是一位专业的康复训练专家，擅长根据用户的身体状况和训练历史提供建议"},
                    {"role": "user", "content": input_text}
                ],
                result_format='message'
            )
            return response.output.choices[0].message.content

        except Exception as e:
            return f"API调用失败: {str(e)}"

    def update_suggestion_label(self, text):
        """更新建议标签内容（需要在主线程调用）"""
        if self.suggestion_label:  # 检查对象是否已存在
            self.suggestion_label.setText(text)
            self.suggestion_label.setStyleSheet("""
                QLabel {
                    font: 14px 'Segoe UI';
                    color: #333333;
                    padding: 10px;
                    background: white;
                    border-radius: 6px;
                    margin: 5px;
                }
            """)
    #此方法正确
    def load_user_profile(self):
        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.age, p.gender, p.height, p.weight, p.condition, u.username
            FROM PatientProfiles p
            JOIN Users u ON p.user_id = u.user_id
            WHERE u.user_id = ?
        """, (self.user_id,))
        profile = cursor.fetchone()
        conn.close()
        print("这是AI建议获取数据")
        print(profile)
        print("AI建议打印完毕")

        if profile:
            return {
                "age": profile[0],
                "gender": profile[1],
                "height": profile[2],
                "weight": profile[3],
                "condition": profile[4],
                "username": profile[5]
            }
        return {}

    def load_training_history_forAI(self):
        # 纯数据获取部分
        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT t.task_description, tr.reps_completed, tr.score, tr.time
                FROM RehabTasks t
                JOIN TrainingRecords tr ON t.task_id = tr.task_id
                WHERE t.patient_id = ?
                ORDER BY tr.time ASC
            """, (self.user_id,))

            training_history = cursor.fetchall()
            print("[DEBUG] 原始数据库查询结果:", training_history)  # 关键调试点

        except sqlite3.Error as e:
            print("数据库查询错误:", e)
            training_history = []
        finally:
            conn.close()

        return training_history

    def load_training_history(self):
        # 连接到SQLite数据库'rehab.db'
        conn = sqlite3.connect('rehab.db')
        cursor = conn.cursor()

        # 修改 TrainingHistoryDialog 类中的 load_training_history 方法
        cursor.execute("""
            SELECT t.task_description, tr.reps_completed, tr.score, datetime(tr.time, '+8 hours') as adjusted_time
            FROM RehabTasks t
            JOIN TrainingRecords tr ON t.task_id = tr.task_id
            WHERE t.patient_id = ?
            ORDER BY adjusted_time ASC
        """, (self.user_id,))

        training_history = cursor.fetchall()
        print(training_history)
        conn.close()

        self.history_table.setRowCount(len(training_history))
        scores = []
        times = []
        for row, record in enumerate(training_history):
            task_desc = record[0]
            completed_reps = record[1]
            score = record[2]
            time = record[3]

            self.history_table.setItem(row, 0, QTableWidgetItem(task_desc))
            self.history_table.setItem(row, 1, QTableWidgetItem(str(completed_reps)))
            self.history_table.setItem(row, 2, QTableWidgetItem(str(score)))
            self.history_table.setItem(row, 3, QTableWidgetItem(time))

            scores.append(score)
            times.append(time)

            # 增强的图表样式
            self.ax.clear()
            self.ax.plot(times, scores,
                         marker='o',
                         color='#4dabf7',
                         markersize=8,
                         markerfacecolor='white',
                         markeredgewidth=2,
                         linewidth=2.5,
                         linestyle='--')

            # 设置现代图表样式
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['left'].set_color('#adb5bd')
            self.ax.spines['bottom'].set_color('#adb5bd')

            self.ax.tick_params(axis='both', colors='#6c757d')
            self.ax.set_xlabel('日期时间', color='#495057', fontsize=10)
            self.ax.set_ylabel('训练评分', color='#495057', fontsize=10)
            self.ax.set_title('历史训练趋势分析',
                              pad=20,
                              fontdict={'fontsize': 14, 'color': '#343a40'})

            # 添加数据标签
            for x, y in zip(times, scores):
                self.ax.annotate(f'{y}分',
                                 (x, y),
                                 textcoords="offset points",
                                 xytext=(0, 10),
                                 ha='center',
                                 fontsize=8,
                                 color='#4dabf7')

            # 设置网格样式
            self.ax.grid(True, alpha=0.3, linestyle='--')

            # 优化时间轴显示
            self.figure.autofmt_xdate(rotation=45, ha='right')
            self.canvas.draw()

    def format_training_history(self, history):
        """格式化训练历史为可读文本"""
        if not history:
            return "暂无训练记录"

        formatted = []
        for record in history:
            task = record[0]
            reps = record[1]
            score = record[2]
            date = record[3]

            formatted.append(f"- {date} | {task} | {reps}次 | 评分：{score:.1f}")

        return "\n".join(formatted)


if __name__ == "__main__":
    init_database()
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())
