import csv
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time
from datetime import datetime

# 导入当前项目的模块
from src.pose_analyzer import PoseAnalyzer

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import matplotlib.font_manager as fm


class PoseMainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("姿态分析系统")

        # 获取屏幕尺寸并设置窗口大小为屏幕的90%
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)

        # 计算窗口居中位置
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1000, 600)  # 设置最小窗口大小

        # 创建姿态分析器对象
        self.pose_analyzer = PoseAnalyzer()

        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建左侧导航栏
        self.create_sidebar()

        # 创建右侧内容区域
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 初始化所有页面
        self.pages = {}
        self.create_pages()

        # 默认显示识别页面
        self.show_page("recognition")

        # 摄像头相关变量
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.pose_data = None
        self.analysis_history = []

        # 启动视频更新
        self.update_video()

    def create_sidebar(self):
        # 创建左侧导航栏
        sidebar = ttk.Frame(self.main_frame, style="Sidebar.TFrame")
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 创建导航按钮
        buttons = [
            ("姿态识别", "recognition"),
            ("数据分析", "analysis"),
        ]

        for text, page in buttons:
            btn = ttk.Button(
                sidebar,
                text=text,
                command=lambda p=page: self.show_page(p),
                style="Sidebar.TButton"
            )
            btn.pack(fill=tk.X, padx=5, pady=2)

    def create_pages(self):
        # 创建识别页面
        recognition_frame = ttk.Frame(self.content_frame)
        self.create_recognition_page(recognition_frame)
        self.pages["recognition"] = recognition_frame

        # 创建分析页面
        analysis_frame = ttk.Frame(self.content_frame)
        self.create_analysis_page(analysis_frame)
        self.pages["analysis"] = analysis_frame

    def show_page(self, page_name):
        # 隐藏所有页面
        for page in self.pages.values():
            page.pack_forget()

        # 显示选中的页面
        self.pages[page_name].pack(fill=tk.BOTH, expand=True)

    def create_recognition_page(self, parent):
        # 创建主框架
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建左右分栏
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # 左侧视频和控制区域
        left_frame = ttk.Frame(paned)

        # 创建顶部控制栏
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 控制按钮
        self.start_btn = ttk.Button(control_frame, text="开始识别", command=self.start_recognition)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(control_frame, text="停止识别", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 视频显示区域
        video_frame = ttk.Frame(left_frame)
        self.video_label = ttk.Label(video_frame, background='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 状态显示区域
        status_frame = ttk.LabelFrame(left_frame, text="当前状态")
        status_frame.pack(fill=tk.X, pady=5)

        # 创建状态信息框架
        status_info_frame = ttk.Frame(status_frame)
        status_info_frame.pack(fill=tk.X, padx=5, pady=5)

        # 添加姿态状态标签
        self.status_label = ttk.Label(status_info_frame, text="等待开始...")
        self.status_label.pack(side=tk.LEFT, padx=5)

        # 将左侧框架添加到分栏
        paned.add(left_frame, weight=3)

        # 右侧姿态历史记录区域
        right_frame = ttk.Frame(paned)

        # 创建历史记录标题和控制按钮
        history_header = ttk.Frame(right_frame)
        history_header.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(history_header, text="姿态历史记录", font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        ttk.Button(history_header, text="清空记录", command=self.clear_pose_history).pack(side=tk.RIGHT)

        # 创建历史记录显示区域（使用Treeview）
        self.history_tree = ttk.Treeview(right_frame, columns=('time', 'pose', 'confidence'), show='headings')
        self.history_tree.heading('time', text='时间')
        self.history_tree.heading('pose', text='姿态')
        self.history_tree.heading('confidence', text='置信度')

        # 设置列宽
        self.history_tree.column('time', width=80)
        self.history_tree.column('pose', width=100)
        self.history_tree.column('confidence', width=80)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)

        # 放置Treeview和滚动条
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # 将右侧框架添加到分栏
        paned.add(right_frame, weight=1)

        # 添加提示信息
        tip_frame = ttk.LabelFrame(parent, text="💡 系统提示", padding=10)
        tip_frame.pack(fill=tk.X, pady=(10, 0))

        tip_text = """
🔥 姿态分析系统功能：
• 实时姿态检测和分析
• 支持站立、坐着、弯腰、躺下、跪姿五种姿态识别
• 提供详细的姿态历史记录和分析
• 基于MediaPipe和深度学习的高精度识别
        """

        tip_label = ttk.Label(tip_frame, text=tip_text, justify=tk.LEFT)
        tip_label.pack(anchor=tk.W)

    def create_analysis_page(self, parent):
        # 创建分析页面的主框架
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建上下分栏
        paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # 上半部分：统计信息
        stats_frame = ttk.LabelFrame(paned, text="姿态统计分析")

        # 统计信息显示区域
        stats_info_frame = ttk.Frame(stats_frame)
        stats_info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建统计标签
        self.stats_labels = {}
        poses = ['standing', 'Sit', 'stoop', 'lying', 'kneel']
        pose_names = ['站立', '坐着', '弯腰', '躺下', '跪姿']

        for i, (pose, name) in enumerate(zip(poses, pose_names)):
            frame = ttk.Frame(stats_info_frame)
            frame.grid(row=i//2, column=i%2, padx=20, pady=10, sticky='w')

            ttk.Label(frame, text=f"{name}:", font=('Arial', 12)).pack(side=tk.LEFT)
            label = ttk.Label(frame, text="0次 (0%)", font=('Arial', 12, 'bold'))
            label.pack(side=tk.LEFT, padx=10)
            self.stats_labels[pose] = label

        # 刷新按钮
        refresh_btn = ttk.Button(stats_frame, text="刷新统计", command=self.refresh_statistics)
        refresh_btn.pack(pady=10)

        paned.add(stats_frame, weight=1)

        # 下半部分：详细记录
        detail_frame = ttk.LabelFrame(paned, text="详细记录")

        # 创建详细记录表格
        self.detail_tree = ttk.Treeview(detail_frame, columns=('id', 'time', 'pose', 'duration'), show='headings')
        self.detail_tree.heading('id', text='序号')
        self.detail_tree.heading('time', text='时间')
        self.detail_tree.heading('pose', text='姿态')
        self.detail_tree.heading('duration', text='持续时间')

        # 设置列宽
        self.detail_tree.column('id', width=50)
        self.detail_tree.column('time', width=120)
        self.detail_tree.column('pose', width=80)
        self.detail_tree.column('duration', width=100)

        # 添加滚动条
        detail_scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail_tree.yview)
        self.detail_tree.configure(yscrollcommand=detail_scrollbar.set)

        # 放置表格和滚动条
        self.detail_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        detail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        paned.add(detail_frame, weight=2)

    def start_recognition(self):
        """开始姿态识别"""
        try:
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开摄像头")
                return

            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="正在识别...")

        except Exception as e:
            messagebox.showerror("错误", f"启动摄像头失败: {str(e)}")

    def stop_recognition(self):
        """停止姿态识别"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="已停止")

    def update_video(self):
        """更新视频显示"""
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # 调整帧大小
                height, width = frame.shape[:2]
                display_width = 640
                display_height = int(height * display_width / width)
                frame = cv2.resize(frame, (display_width, display_height))
                frame = cv2.flip(frame, 1)
                # 使用姿态分析器处理帧
                try:
                    processed_frame, pose_data = self.pose_analyzer.process_frame(frame)
                    self.current_frame = processed_frame
                    self.pose_data = pose_data

                    # 更新状态显示
                    if pose_data and pose_data.get('pose'):
                        pose_name = pose_data['pose']
                        pose_names = {
                            'standing': '站立',
                            'Sit': '坐着',
                            'stoop': '弯腰',
                            'lying': '躺下',
                            'kneel': '跪姿'
                        }
                        display_name = pose_names.get(pose_name, pose_name)
                        self.status_label.config(text=f"检测到姿态: {display_name}")

                        # 添加到历史记录
                        self.add_pose_to_history(pose_name)

                    # 转换为PIL图像并显示
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(image)

                    self.video_label.configure(image=photo)
                    self.video_label.image = photo  # 保持引用

                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    # 如果处理失败，显示原始帧
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(image)
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo

        # 每30ms更新一次
        self.root.after(30, self.update_video)

    def add_pose_to_history(self, pose_name):
        """添加姿态到历史记录"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 添加到历史记录列表
        self.analysis_history.append({
            'time': timestamp,
            'pose': pose_name,
            'confidence': 0.95  # 这里可以从实际分析结果获取置信度
        })

        # 更新历史记录显示（只显示最近20条）
        self.update_history_display()

    def update_history_display(self):
        """更新历史记录显示"""
        # 清空当前显示
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        # 显示最近20条记录
        recent_history = self.analysis_history[-20:]
        pose_names = {
            'standing': '站立',
            'Sit': '坐着',
            'stoop': '弯腰',
            'lying': '躺下',
            'kneel': '跪姿'
        }

        for record in recent_history:
            display_name = pose_names.get(record['pose'], record['pose'])
            self.history_tree.insert('', 0, values=(
                record['time'],
                display_name,
                f"{record['confidence']:.2f}"
            ))

    def clear_pose_history(self):
        """清空姿态历史记录"""
        self.analysis_history.clear()
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for item in self.detail_tree.get_children():
            self.detail_tree.delete(item)

    def refresh_statistics(self):
        """刷新统计信息"""
        if not self.analysis_history:
            # 如果没有历史记录，显示0
            for label in self.stats_labels.values():
                label.config(text="0次 (0%)")
            return

        # 统计各种姿态的次数
        pose_counts = {}
        for record in self.analysis_history:
            pose = record['pose']
            pose_counts[pose] = pose_counts.get(pose, 0) + 1

        total_count = len(self.analysis_history)

        # 更新统计显示
        pose_names = {
            'standing': '站立',
            'Sit': '坐着',
            'stoop': '弯腰',
            'lying': '躺下',
            'kneel': '跪姿'
        }

        for pose, label in self.stats_labels.items():
            count = pose_counts.get(pose, 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            label.config(text=f"{count}次 ({percentage:.1f}%)")

        # 更新详细记录
        self.update_detail_records()

    def update_detail_records(self):
        """更新详细记录显示"""
        # 清空当前显示
        for item in self.detail_tree.get_children():
            self.detail_tree.delete(item)

        # 处理历史记录，计算持续时间
        if not self.analysis_history:
            return

        pose_names = {
            'standing': '站立',
            'Sit': '坐着',
            'stoop': '弯腰',
            'lying': '躺下',
            'kneel': '跪资'
        }

        # 简化处理：显示所有记录
        for i, record in enumerate(self.analysis_history):
            display_name = pose_names.get(record['pose'], record['pose'])
            self.detail_tree.insert('', 'end', values=(
                i + 1,
                record['time'],
                display_name,
                "1秒"  # 简化显示
            ))


def main():
    root = tk.Tk()
    # 设置主题样式
    style = ttk.Style()
    style.configure("Sidebar.TFrame", background="#2c3e50")
    style.configure("Sidebar.TButton",
                   background="#2c3e50",
                   foreground="black",
                   padding=10)

    app = PoseMainApplication(root)

    # 绑定窗口关闭事件
    def on_closing():
        if app.is_running:
            app.stop_recognition()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()