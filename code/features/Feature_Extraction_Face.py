import numpy as np
from typing import List, Dict, Tuple, Callable
from math import isnan
import utils

def calculate_mar(landmarks: List[List[float]]) -> float:
    """计算嘴部纵横比(Mouth Aspect Ratio)"""
    try:
        if len(landmarks) < 68:
            return 0.0
        
        points = landmarks
        
        mouth_width = np.linalg.norm(np.array(points[54]) - np.array(points[48]))
        left_lip = np.linalg.norm(np.array(points[58]) - np.array(points[50]))
        right_lip = np.linalg.norm(np.array(points[56]) - np.array(points[52]))
        
        if mouth_width < 1e-6:
            return 0.0
            
        mar = (left_lip + right_lip) / (2 * mouth_width)
        return round(float(mar), 4) if not isnan(mar) else 0.0
    except Exception as e:
        return 0.0

def calculate_ear(landmarks: List[List[float]]) -> Tuple[float, float]:
    """计算眼睛纵横比(Eye Aspect Ratio)，返回左眼和右眼的EAR"""
    try:
        if len(landmarks) < 68:
            return (0.0, 0.0)
        
        # 左眼关键点 (36-41)
        left_eye = [
            landmarks[36], landmarks[37], 
            landmarks[38], landmarks[39],
            landmarks[40], landmarks[41]
        ]
        
        # 右眼关键点 (42-47)
        right_eye = [
            landmarks[42], landmarks[43],
            landmarks[44], landmarks[45],
            landmarks[46], landmarks[47]
        ]
        
        # 计算左眼EAR
        A_left = np.linalg.norm(np.array(left_eye[1]) - np.array(left_eye[5]))
        B_left = np.linalg.norm(np.array(left_eye[2]) - np.array(left_eye[4]))
        C_left = np.linalg.norm(np.array(left_eye[0]) - np.array(left_eye[3]))
        ear_left = (A_left + B_left) / (2 * C_left) if C_left > 1e-6 else 0.0
        
        # 计算右眼EAR
        A_right = np.linalg.norm(np.array(right_eye[1]) - np.array(right_eye[5]))
        B_right = np.linalg.norm(np.array(right_eye[2]) - np.array(right_eye[4]))
        C_right = np.linalg.norm(np.array(right_eye[0]) - np.array(right_eye[3]))
        ear_right = (A_right + B_right) / (2 * C_right) if C_right > 1e-6 else 0.0
        
        ear_left = round(float(ear_left), 4) if not isnan(ear_left) else 0.0
        ear_right = round(float(ear_right), 4) if not isnan(ear_right) else 0.0
        
        return (ear_left, ear_right)
    except Exception as e:
        return (0.0, 0.0)

def get_facial_feature_calculators() -> List[Tuple[str, Callable]]:
    """返回全脸特征计算函数列表（优化版）"""
    # 定义特征组和对应的统计函数
    feature_groups = [
        ('mar', 'mars'),
        ('left_ear', 'left_ears'),
        ('right_ear', 'right_ears'),
    ]
    
    # 定义统计操作
    stats = [
        ('mean', np.nanmean),
        ('median', np.nanmedian),
        ('std', np.nanstd),
        ('min', np.nanmin),
        ('max', np.nanmax)
    ]
    
    # 动态生成特征计算器列表
    calculators = []
    for prefix, key in feature_groups:
        for stat_name, stat_func in stats:
            feature_name = f"{prefix}_{stat_name}"
            calculators.append((feature_name, lambda x, f=stat_func, k=key: f(x[k])))
    
    return calculators

def extract_facial_features(window_data: List[Dict]) -> Dict[str, np.ndarray]:
    """从窗口数据中提取全脸特征"""
    landmarks_list = [frame.get('landmarks', []) for frame in window_data]
    
    # 计算各帧的基础特征
    mars = np.array([calculate_mar(landmarks) for landmarks in landmarks_list])
    
    # 计算各帧的眼睛纵横比
    ears = [calculate_ear(landmarks) for landmarks in landmarks_list]
    left_ears = np.array([ear[0] for ear in ears])
    right_ears = np.array([ear[1] for ear in ears])


    return {
        'mars': mars,
        'left_ears': left_ears,
        'right_ears': right_ears,
    }

def facial_features(input_path: str, window_size_sec: float, step_size_sec: float, label_path: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """从输入文件中提取全脸特征（优化版，自动丢弃不足窗口）"""
    try:
        # 数据加载
        data = utils.load_and_validate_json(input_path)
        fps = float(data['video_info']['fps'])
        frames = data.get('frames', [])
        
        # 时间范围处理
        min_start_time, max_end_time = utils.get_valid_time_range(label_path)
        frames = utils.crop_data_by_time(frames, fps, min_start_time, max_end_time)
        
        # 窗口处理（只处理完整窗口）
        window_size, step_size = utils.calculate_window_params(fps, window_size_sec, step_size_sec)
        valid_indices = np.arange(0, len(frames) - window_size + 1, step_size)
        
        # 特征计算器
        feature_calculators = get_facial_feature_calculators()
        feature_names = [name for name, _ in feature_calculators]
        
        # 预分配数组
        features = np.zeros((len(valid_indices), len(feature_calculators)))
        metadata = []
        
        # 处理每个有效窗口
        for i, start in enumerate(valid_indices):
            end = start + window_size
            
            # 记录元数据（使用绝对时间）
            metadata.append(utils.create_window_metadata(
                min_start_time * fps + start,
                min_start_time * fps + end,
                fps
            ))
            
            # 向量化特征计算
            window_data = frames[start:end]
            feature_dict = extract_facial_features(window_data)
            features[i] = [func(feature_dict) for _, func in feature_calculators]
        
        return features, feature_names, metadata
    
    except Exception as e:
        print(f"面部特征提取错误: {str(e)}")
        n_features = len(get_facial_feature_calculators())
        return np.zeros((0, n_features)), [name for name, _ in get_facial_feature_calculators()], []
    

# 使用示例
if __name__ == "__main__":
    prefix = '/data/Leo/mm/data/Newborn200/'
    input_json = prefix + "Face/50cm3.24kg_face_landmarks.json"
    label_file = prefix + "Label/50cm3.24kg.txt"    
    
    features, names, meta = facial_features(
        input_path=input_json,
        window_size_sec=2.5,
        step_size_sec=2.5,
        label_path=label_file
    )
    print(f"特征矩阵形状: {features.shape}")
    print(f"特征名称: {names}")
    print(f"窗口元数据: {meta[:5]}")  # 打印前5个窗口的元数据