import os
import random
import numpy as np
import torch
from typing import Dict, Tuple, Union
import json

def initialize_random_seed(seed):
    """初始化随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def prepare_feature_output_path(dataDir, feature_type):
    """准备特征输出路径"""
    base_name = os.path.splitext(os.path.basename(dataDir))[0]
    parent_dir = os.path.dirname(dataDir)
    new_parent = os.path.join(os.path.dirname(parent_dir), feature_type)
    os.makedirs(new_parent, exist_ok=True)
    return base_name, new_parent

def get_label_file_path(dataDir):
    """获取标签文件路径"""
    base_name, new_parent = prepare_feature_output_path(dataDir, "Label")
    return os.path.join(new_parent, f"{base_name}.txt")

def get_motion_feature_file_path(dataDir):
    """获取运动特征文件路径"""
    base_name, new_parent = prepare_feature_output_path(dataDir, "Body")
    return os.path.join(new_parent, f"{base_name}_motion_features.json")

def get_face_landmark_file_path(dataDir):
    """获取面部特征文件路径"""
    base_name, new_parent = prepare_feature_output_path(dataDir, "Face")
    return os.path.join(new_parent, f"{base_name}_face_landmarks.json")

def extract_subject_id(filepath):
    """提取受试者ID"""
    return os.path.splitext(os.path.basename(filepath))[0]
    
def generate_feature_filename(label_path):
    """生成特征文件名"""
    basename = os.path.splitext(os.path.basename(label_path))[0]
    return f"{basename}_features.npz"

# utils.py

def load_and_validate_json(input_path: str) -> dict:
    """加载并验证JSON文件"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    with open(input_path) as f:
        return json.load(f)

def get_valid_time_range(label_path: str) -> Tuple[float, float]:
    """从标签文件中获取有效时间范围"""
    min_start_time, max_end_time = float('inf'), 0.0
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                if len(parts := line.strip().split('\t')) >= 2:
                    min_start_time = min(min_start_time, float(parts[0]))
                    max_end_time = max(max_end_time, float(parts[1]))
    return min_start_time, max_end_time

def crop_data_by_time(data: list, fps: float, min_start_time: float, max_end_time: float) -> list:
    """根据时间范围裁剪数据"""
    start_frame = int(min_start_time * fps)
    end_frame = int(max_end_time * fps)
    return data[start_frame:end_frame]

def calculate_window_params(fps: float, window_size_sec: float, step_size_sec: float) -> Tuple[int, int]:
    """计算窗口参数"""
    return (
        int(round(window_size_sec * fps)),
        int(round(step_size_sec * fps))
    )

def create_window_metadata(start_idx: int, end_idx: int, fps: float) -> Dict:
    """创建窗口元数据"""
    return {
        'start_frame': start_idx,
        'end_frame': end_idx - 1,
        'start_sec': start_idx / fps,
        'end_sec': (end_idx - 1) / fps,
    }