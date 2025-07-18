import numpy as np
import utils
from typing import List, Dict, Tuple, Callable
import warnings
import config
import numpy as np

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)

def extract_motion_components(window_data: List[Dict], key: str) -> Dict[str, np.ndarray]:
    """从窗口数据中提取x、y、r、角度四个维度的数据"""
    components = {'x': [], 'y': [], 'r': [], 'angle': []}
    for frame in window_data:
        if key in frame and frame[key]:
            best = max(frame[key], key=lambda x: x[4])
            components['x'].append(best[0])
            components['y'].append(best[1])
            components['r'].append(best[2])
            components['angle'].append(best[3])
        else:
            # 填充默认值（例如 0）
            components['x'].append(0)
            components['y'].append(0)
            components['r'].append(0)
            components['angle'].append(0)

    return {k: np.nan_to_num(np.array(v), nan=0) for k, v in components.items()}

def get_feature_calculators(fps: int) -> List[Tuple[str, Callable]]:
    """返回特征计算函数列表（多维度优化版）"""
    return [
        # 基础统计量
        ('std', lambda d: np.std(d['r'])),
        ('median', lambda d: np.median(d['r'])),
        ('mean', lambda d: np.mean(d['r'])),
        ('max', lambda d: np.max(d['r'])),
        ('min', lambda d: np.min(d['r'])),
    ]

def body_features(input_path: str, window_size_sec: float, step_size_sec: float, label_path: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """从输入文件中提取运动特征（优化版，自动丢弃不足窗口）"""
    # 加载数据
    data = utils.load_and_validate_json(input_path)
    features = data['features']
    fps = data['video_info']['fps']
    
    # 获取有效时间范围并裁剪数据
    min_start_time, max_end_time = utils.get_valid_time_range(label_path)
    features = utils.crop_data_by_time(features, fps, min_start_time, max_end_time)
    
    # 计算窗口参数并生成有效窗口索引
    window_size, step_size = utils.calculate_window_params(fps, window_size_sec, step_size_sec)
    valid_indices = np.arange(0, len(features) - window_size + 1, step_size)
    
    # 获取特征计算器
    feature_calculators = get_feature_calculators(fps)
    feature_keys = ['Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'] # 局部肢体
    # feature_keys = ['WholeBody'] # 人体整体
    # feature_keys = ['WholeFrameMotion'] # 完整画面

    feature_names = [
        f"{key}_{name}" for key in feature_keys 
        for name, _ in feature_calculators
    ]
    
    # 预分配数组（使用有效窗口数量）
    features_array = np.zeros((len(valid_indices), len(feature_names)))
    window_metadata = []
    
    # 向量化处理每个有效窗口
    for i, start_idx in enumerate(valid_indices):
        end_idx = start_idx + window_size
        
        # 记录元数据（自动计算绝对时间）
        window_metadata.append(utils.create_window_metadata(
            min_start_time * fps + start_idx,
            min_start_time * fps + end_idx,
            fps
        ))
        
        # 计算特征（向量化版本）
        window_data = features[start_idx:end_idx]
        feature_values = []
        for key in feature_keys:
            motion = extract_motion_components(window_data, key)
            feature_values.extend(
                calculator(motion) 
                for _, calculator in feature_calculators
            )
        
        features_array[i] = feature_values
    
    return features_array, feature_names, window_metadata

def extract_raw_motion_features(dataDir):
    file = utils.get_motion_feature_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    
    data = utils.load_and_validate_json(file)
    features = data['features']  
    fps = data['video_info']['fps']
    
    min_start_time, max_end_time = utils.get_valid_time_range(labelDir)
    features = utils.crop_data_by_time(features, fps, min_start_time, max_end_time)
    
    window_size, step_size = utils.calculate_window_params(fps, config.slidingWindows, config.step)
    valid_indices = np.arange(0, len(features) - window_size + 1, step_size)
    
    feature_names = ['Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg']
    feature_List = []
    for i, start_idx in enumerate(valid_indices):
        end_idx = start_idx + window_size

        window_data = features[start_idx:end_idx]
        feature_values = []
        for name in feature_names:
            motion = extract_motion_components(window_data, name)['r']       
            feature_values.append(motion) 

        feature_List.append(np.array(feature_values))
    motion_features = np.array(feature_List)

    return motion_features, feature_names
