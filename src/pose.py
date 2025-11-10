import numpy as np
import tensorflow as tf
import os
from pathlib import Path

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path=None,
        num_threads=1,
    ):
        # 如果没有指定模型路径，使用默认路径
        if model_path is None:
            # 获取项目根目录（src 的父目录）
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            model_path = str(project_root / 'model' / 'pose_model.tflite')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建解释器
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        
        # 分配张量
        self.interpreter.allocate_tensors()
        
        # 获取输入输出细节
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 预热模型
        self.interpreter.set_tensor(self.input_details[0]['index'], np.zeros(self.input_details[0]['shape'], dtype=np.float32))
        self.interpreter.invoke()

    def __call__(self,landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))
        
        return result_index