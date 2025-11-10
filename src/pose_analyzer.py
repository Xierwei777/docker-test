import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict, Optional
import tensorflow as tf
from src.pose import KeyPointClassifier
import time
# import copy  # 🔥 优化：移除copy模块，使用numpy.copy更快
import os
from datetime import datetime
from src.pose_logger import PoseLogger
class PoseAnalyzer:
    def __init__(self, model_complexity=1, enable_gpu=True, enable_logging=True, console_output=True, record_interval=1.0):
        """
        初始化姿态分析器
        
        Args:
            model_complexity: 模型复杂度
                0 = Lite (最快，精度较低)
                1 = Full (平衡，推荐) 
                2 = Heavy (最慢，精度最高)
            enable_gpu: 是否启用GPU加速
            enable_logging: 是否启用日志记录
            console_output: 是否在控制台打印检测记录（打印台风格）
            record_interval: 日志记录间隔（秒），不影响视频帧率
        """
        # 配置 GPU
        if enable_gpu:
            self._configure_gpu()
        
        # 初始化日志记录器
        self.logger = PoseLogger(console_output=console_output, record_interval=record_interval) if enable_logging else None
        self.enable_logging = enable_logging
        
        # 初始化MediaPipe姿态检测
        self.mp_pose = mp.solutions.pose
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(f"[OK] MediaPipe Pose initialized (model complexity: {model_complexity})")
        except Exception as e:
            print(f"[ERROR] Failed to initialize MediaPipe: {e}")
            print("[TIP] Trying Lite model (model_complexity=0)")
            # 降级到 Lite 模型
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[OK] Downgraded to Lite model")
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 初始化姿态分类器（只创建一次，避免重复加载模型）
        try:
            self.classifier = KeyPointClassifier()
            print("[OK] Pose classifier initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize pose classifier: {e}")
            self.classifier = None
        
        # 姿态类别定义
        self.dir=['standing','Sit','stoop','lying','kneel']

        # 添加绘制相关属性
        self.current_pose = None
        self.last_pose = None
        self.pose_count = 0
        self.pose_start_time = None
        self.use_brect = True
    
    def _configure_gpu(self):
        """配置 GPU 加速（支持双显卡笔记本）"""
        try:
            # 添加 CUDA DLL 路径（Windows）
            if os.name == 'nt':  # Windows
                cuda_paths = [
                    'C:/Program Files/NVIDIA/cuda/bin',
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin',
                ]
                for path in cuda_paths:
                    if os.path.exists(path):
                        try:
                            os.add_dll_directory(path)
                            print(f"[GPU] 已添加 DLL 路径: {path}")
                        except Exception as e:
                            pass
            
            # 检测所有 GPU
            all_gpus = tf.config.list_physical_devices('GPU')
            
            if all_gpus:
                print(f"[GPU] Detected {len(all_gpus)} GPU device(s):")
                for i, gpu in enumerate(all_gpus):
                    print(f"[GPU] GPU {i}: {gpu.name}")
                
                # 过滤出 NVIDIA GPU（跳过 Intel 核显）
                # Intel GPU 通常包含 "Intel" 或 "HD Graphics"
                # NVIDIA GPU 通常包含 "NVIDIA" 或 "GeForce"
                nvidia_gpus = []
                for i, gpu in enumerate(all_gpus):
                    gpu_name = gpu.name.lower()
                    # 跳过 Intel 核显
                    if 'intel' in gpu_name or 'hd graphics' in gpu_name:
                        print(f"[GPU] Skipping GPU {i} (Intel integrated)")
                        continue
                    # 使用 NVIDIA GPU
                    nvidia_gpus.append(gpu)
                    print(f"[GPU] Selected GPU {i} (NVIDIA discrete) [OK]")
                
                if nvidia_gpus:
                    # 为 NVIDIA GPU 启用内存增长
                    for gpu in nvidia_gpus:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        except RuntimeError as e:
                            print(f"[GPU] Memory configuration failed: {e}")
                    
                    # 只设置 NVIDIA GPU 为可见设备
                    tf.config.set_visible_devices(nvidia_gpus, 'GPU')
                    
                    print(f"[GPU] Enabled {len(nvidia_gpus)} NVIDIA GPU(s)")
                    
                    # 设置 TensorFlow 日志级别
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                    
                else:
                    print("[GPU] No NVIDIA GPU detected, using CPU mode")
                    # CPU 优化
                    tf.config.threading.set_intra_op_parallelism_threads(4)
                    tf.config.threading.set_inter_op_parallelism_threads(4)
                
            else:
                print("[GPU] No GPU detected, using CPU mode")
                # CPU 优化
                tf.config.threading.set_intra_op_parallelism_threads(4)
                tf.config.threading.set_inter_op_parallelism_threads(4)
                
        except Exception as e:
            print(f"[GPU] Configuration failed: {e}")
            print("[GPU] Using CPU mode")
            # CPU 优化作为后备方案
            try:
                tf.config.threading.set_intra_op_parallelism_threads(4)
                tf.config.threading.set_inter_op_parallelism_threads(4)
            except:
                pass
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        处理单帧图像，返回带有姿态标注的图像和姿态数据
        """
        # 生成时间戳
        timestamp = time.time()
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 转换颜色空间
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        # 🔥 优化：使用numpy.copy替代copy.deepcopy，速度快3-5倍
        debug_image = frame.copy()

        # 提取关键点
        pose_landmarks = []
        pose_result = None
        has_target = False
        confidence = 0.0

        if results.pose_landmarks:
            has_target = True
            for landmark in results.pose_landmarks.landmark:
                pose_landmarks.append([landmark.x, landmark.y])

            # 计算边界框
            brect = self.calc_bounding_rect(debug_image, pose_landmarks)
            # 预处理关键点用于姿态分析
            pose_landmarks_flat = [landmark[i] for landmark in pose_landmarks for i in range(2)]
            pose_result = self.analyze_pose_sequence(pose_landmarks_flat)
            
            # 计算置信度（基于关键点的可见性）
            confidence = sum([lm.visibility for lm in results.pose_landmarks.landmark]) / len(results.pose_landmarks.landmark)

            # 绘制部分 - 使用参考文件的绘制方法
            debug_image = self.draw_bounding_rect(self.use_brect, debug_image, brect)
            self.mp_drawing.draw_landmarks(
                debug_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            debug_image = self.draw_pose_info_text(debug_image, brect, pose_result)
        else:
            # 没有检测到目标
            debug_image = self.draw_no_target_info(debug_image)
        
        # 构建分析结果
        analysis_result = {
            'timestamp': timestamp,
            'datetime': datetime_str,
            'landmarks': pose_landmarks,
            'pose': pose_result if pose_result else 'No Target',
            'has_target': has_target,
            'landmarks_count': len(pose_landmarks),
            'confidence': round(confidence, 3),
            'note': '正常检测' if has_target else '未检测到目标'
        }
        
        # 记录分析结果
        if self.enable_logging and self.logger:
            self.logger.log_analysis(analysis_result)
        
        return debug_image, analysis_result
    
    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # 外接矩形
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

        return image
    #计算边框
    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks):
            landmark_x = min(int(landmark[0] * image_width), image_width - 1)
            landmark_y = min(int(landmark[1] * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def analyze_pose_sequence(self, landmarks_sequence: List[List[float]]) -> str:
        """
        分析姿态序列，识别动作类型
        """
        if self.classifier is None:
            return "Unknown"
        
        try:
            result_index = self.classifier(landmarks_sequence)
            return self.dir[result_index]
        except Exception as e:
            print(f"[ERROR] Pose classification failed: {e}")
            return "Unknown"

    def draw_pose_info_text(self, image, brect, pose_text):
        """绘制姿态信息文本"""
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

        info_text = f"Pose: {pose_text}" if pose_text else "Pose: Unknown"

        # 更新当前姿态
        if self.current_pose != pose_text:
            self.last_pose = self.current_pose
            self.current_pose = pose_text
            self.pose_count = 1
            self.pose_start_time = time.time()
        else:
            self.pose_count += 1

        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 左上角详细数据已移除（用户不需要）
        # - Duration（持续时间）
        # - Time（时间戳）
        # 这些数据仍会在日志监控中显示

        return image
    
    def draw_no_target_info(self, image):
        """当没有检测到目标时显示提示信息"""
        height, width = image.shape[:2]
        
        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
        
        # 显示"无目标"提示
        no_target_text = "No Target Detected"
        no_target_text_cn = "未检测到目标"
        
        cv2.putText(image, no_target_text, (width // 2 - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(image, no_target_text, (width // 2 - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 中文提示（使用简单符号代替，因为OpenCV中文显示较复杂）
        hint_text = "Please stand in front of the camera"
        cv2.putText(image, hint_text, (width // 2 - 200, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 左上角时间戳已移除（用户不需要）
        
        return image

    def draw_pose_landmarks(self, image, landmarks):
        """绘制姿态关键点"""
        if not landmarks:
            return image
        
        # 绘制关键点
        for landmark in landmarks:
            if len(landmark) >= 2:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        
        return image

    def start_realtime_analysis(self):
        """
        启动实时姿态分析
        """
        cap=cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            display_width = 1080
            display_height = int(height * display_width / width)

            frame = cv2.resize(frame, (display_width, display_height))
            # 处理帧
            processed_frame, analysis_data = self.process_frame(frame)

            # 显示结果
            cv2.imshow('Pose Analysis', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = PoseAnalyzer()
    analyzer.start_realtime_analysis()