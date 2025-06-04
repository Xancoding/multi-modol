import numpy as np
import json
from typing import List, Dict, Tuple, Callable
import os
from math import isnan

def calculate_mar(landmarks: List[List[float]]) -> float:
    """计算嘴部纵横比(Mouth Aspect Ratio)"""
    try:
        if len(landmarks) < 68:
            return 0.0
        
        # 关键点索引 (48: 左嘴角, 54: 右嘴角, 51: 上唇中点, 57: 下唇中点, 62: 上唇内部, 66: 下唇内部)
        points = {
            48: landmarks[48] if len(landmarks) > 48 else [0, 0],
            51: landmarks[51] if len(landmarks) > 51 else [0, 0],
            54: landmarks[54] if len(landmarks) > 54 else [0, 0],
            57: landmarks[57] if len(landmarks) > 57 else [0, 0],
            62: landmarks[62] if len(landmarks) > 62 else [0, 0],
            66: landmarks[66] if len(landmarks) > 66 else [0, 0]
        }
        
        mouth_width = np.linalg.norm(np.array(points[54]) - np.array(points[48]))
        upper_lip = np.linalg.norm(np.array(points[62]) - np.array(points[51]))
        lower_lip = np.linalg.norm(np.array(points[66]) - np.array(points[57]))
        
        if mouth_width < 1e-6:
            return 0.0
            
        mar = (upper_lip + lower_lip) / (2 * mouth_width)
        return round(float(mar), 4) if not isnan(mar) else 0.0
    except Exception as e:
        return 0.0

def calculate_ear(landmarks: List[List[float]]) -> float:
    """计算眼睛纵横比(Eye Aspect Ratio)"""
    try:
        if len(landmarks) < 68:
            return 0.0
        
        # 左眼关键点 (36-41)
        left_eye = [
            landmarks[36], landmarks[37], 
            landmarks[38], landmarks[39],
            landmarks[40], landmarks[41]
        ]
        
        # 计算左眼EAR
        A = np.linalg.norm(np.array(left_eye[1]) - np.array(left_eye[5]))
        B = np.linalg.norm(np.array(left_eye[2]) - np.array(left_eye[4]))
        C = np.linalg.norm(np.array(left_eye[0]) - np.array(left_eye[3]))
        ear_left = (A + B) / (2 * C) if C > 1e-6 else 0.0
        
        return round(float(ear_left), 4) if not isnan(ear_left) else 0.0
    except Exception as e:
        return 0.0

def calculate_eyebrow_raise(landmarks: List[List[float]]) -> float:
    """计算眉毛上扬程度"""
    try:
        if len(landmarks) < 68:
            return 0.0
        
        # 左眉毛关键点 (17-21)
        eyebrow_points = [landmarks[i] for i in range(17, 22)]
        eye_points = [landmarks[i] for i in range(36, 42)]
        
        # 计算眉毛到眼睛的平均垂直距离
        distances = []
        for i in range(len(eyebrow_points)):
            vertical_dist = eyebrow_points[i][1] - eye_points[i//2][1]
            distances.append(vertical_dist)
        
        return round(float(np.mean(distances)), 4) if distances else 0.0
    except Exception as e:
        return 0.0

def get_facial_feature_calculators() -> List[Tuple[str, Callable]]:
    """返回全脸特征计算函数列表"""
    return [
        # 嘴部特征
        ('mar_mean', lambda x: np.nanmean(x['mars'])),
        ('mar_std', lambda x: np.nanstd(x['mars'])),
        ('mar_min', lambda x: np.nanmin(x['mars'])),
        ('mar_max', lambda x: np.nanmax(x['mars'])),
        # ('mar_velocity', lambda x: np.nanmean(np.abs(np.diff(x['mars'])))),
        
        # 眼部特征
        ('ear_mean', lambda x: np.nanmean(x['ears'])),
        ('ear_std', lambda x: np.nanstd(x['ears'])),
        ('ear_min', lambda x: np.nanmin(x['ears'])),
        ('ear_max', lambda x: np.nanmax(x['ears'])),
        # ('ear_velocity', lambda x: np.nanmean(np.abs(np.diff(x['ears'])))),
        
        # 眉毛特征
        ('eyebrow_mean', lambda x: np.nanmean(x['eyebrows'])),
        ('eyebrow_std', lambda x: np.nanstd(x['eyebrows'])),
        ('eyebrow_min', lambda x: np.nanmin(x['eyebrows'])),
        ('eyebrow_max', lambda x: np.nanmax(x['eyebrows'])),
        # ('eyebrow_velocity', lambda x: np.nanmean(np.abs(np.diff(x['eyebrows'])))),
    ]

def extract_facial_features(window_data: List[Dict]) -> Dict[str, np.ndarray]:
    """从窗口数据中提取全脸特征"""
    landmarks_list = [frame.get('landmarks', []) for frame in window_data]
    
    # 计算各帧的基础特征
    mars = np.array([calculate_mar(landmarks) for landmarks in landmarks_list])
    ears = np.array([calculate_ear(landmarks) for landmarks in landmarks_list])
    eyebrows = np.array([calculate_eyebrow_raise(landmarks) for landmarks in landmarks_list])


    return {
        'mars': mars,
        'ears': ears,
        'eyebrows': eyebrows,
    }

def facial_features(input_path: str, window_size_sec: float, step_size_sec: float, label_path: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """
    从输入文件中提取全脸特征
    
    参数:
        input_path: 输入JSON文件路径
        window_size_sec: 分析窗口大小(秒)
        step_size_sec: 滑动步长(秒)
        label_path: 标签文件路径
        
    返回:
        features_array: 特征矩阵 (n_windows, 12)
        feature_names: 特征名称列表
        window_metadata: 窗口元数据列表
    """
    try:
        # 数据加载与验证
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"面部特征文件不存在: {input_path}")
            
        with open(input_path) as f:
            data = json.load(f)
        
        fps = float(data['video_info']['fps'])
        frames = data.get('frames', [])
        
        # 时间范围处理
        min_start_time, max_end_time = float('inf'), 0.0
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    if len(parts := line.strip().split('\t')) >= 2:
                        min_start_time = min(min_start_time, float(parts[0]))
                        max_end_time = max(max_end_time, float(parts[1]))
        
        # 转换为帧索引并裁剪
        frames = frames[int(min_start_time * fps):int(max_end_time * fps)]
        
        # 窗口处理
        window_size_frames = int(round(window_size_sec * fps))
        step_size_frames = int(round(step_size_sec * fps))
        
        # 特征计算
        feature_calculators = get_facial_feature_calculators()
        feature_names = [name for name, _ in feature_calculators]
        n_windows = len(range(0, len(frames), step_size_frames))
        
        features = np.zeros((n_windows, len(feature_calculators)))
        metadata = []
        
        for i, start in enumerate(range(0, len(frames), step_size_frames)):
            end = min(start + window_size_frames, len(frames))
            window_data = frames[start:end]
            
            metadata.append({
                'start_frame': start,
                'end_frame': end - 1,
                'start_sec': start / fps,
                'end_sec': (end - 1) / fps,
            })
            
            # 提取全脸特征
            feature_dict = extract_facial_features(window_data)
            
            # 计算窗口特征
            features[i] = [func(feature_dict) for _, func in feature_calculators]
        
        return features, feature_names
    
    except Exception as e:
        print(f"面部特征提取错误: {str(e)}")
        return np.zeros((1, len(get_facial_feature_calculators()))), [name for name, _ in get_facial_feature_calculators()], []


# 使用示例
if __name__ == "__main__":
    prefix = '/data/Leo/mm/data/Newborn200/'
    input_json = prefix + "Face/50cm3.24kg_face_landmarks.json"
    label_file = prefix + "Label/50cm3.24kg.txt"    
    
    features, names = facial_features(
        input_path=input_json,
        window_size_sec=2.5,
        step_size_sec=2.5,
        label_path=label_file
    )
    print(f"特征矩阵形状: {features.shape}")
    print(f"特征名称: {names}")
