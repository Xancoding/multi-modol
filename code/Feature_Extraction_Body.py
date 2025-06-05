import numpy as np
import utils
from typing import List, Dict, Tuple, Callable
import warnings

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)

def extract_motion_components(window_data: List[Dict], key: str) -> Dict[str, np.ndarray]:
    """从窗口数据中提取x、y、r、角度四个维度的数据"""
    x, y, r, angle = [], [], [], []
    for frame in window_data:
        if key in frame and frame[key]:
            if isinstance(frame[key][0], (list, np.ndarray)):  # 多目标情况
                for sublist in frame[key]:
                    if len(sublist) >= 4:
                        x.append(sublist[0])
                        y.append(sublist[1])
                        r.append(sublist[2])
                        angle.append(sublist[3])
            elif len(frame[key]) >= 4:  # 单目标情况
                x.append(frame[key][0])
                y.append(frame[key][1])
                r.append(frame[key][2])
                angle.append(frame[key][3])
    
    return {
        'x': np.nan_to_num(np.array(x), nan=0),
        'y': np.nan_to_num(np.array(y), nan=0),
        'r': np.nan_to_num(np.array(r), nan=0),
        'angle': np.nan_to_num(np.array(angle), nan=0)
    }

def calculate_entropy(values: np.ndarray) -> float:
    """计算近似熵（运动不规则性）"""
    clean_values = values[~np.isnan(values)]
    if len(clean_values) < 10:
        return 0
    hist = np.histogram(clean_values, bins=10)[0]
    prob = hist / hist.sum() + 1e-10  # 避免log(0)
    return -np.sum(prob * np.log(prob))


def get_feature_calculators(fps: int) -> List[Tuple[str, Callable]]:
    """返回特征计算函数列表（多维度优化版）"""
    return [
        # 基础统计量
        ('std', lambda d: np.std(d['r'])),
        ('mean', lambda d: np.mean(d['r'])),
        ('max', lambda d: np.max(d['r'])),
        ('min', lambda d: np.min(d['r'])),
        ('entropy', lambda d: calculate_entropy(d['r'])),
    ]

def body_features(input_path: str, window_size_sec: float, step_size_sec: float, label_path: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """从输入文件中提取运动特征"""
    # 加载数据
    data = utils.load_and_validate_json(input_path)
    features = data['features']
    fps = data['video_info']['fps']
    
    # 获取有效时间范围
    min_start_time, max_end_time = utils.get_valid_time_range(label_path)
    features = utils.crop_data_by_time(features, fps, min_start_time, max_end_time)
    
    # 计算窗口参数
    window_size_frames, step_size_frames = utils.calculate_window_params(fps, window_size_sec, step_size_sec)
    
    # 获取特征计算器
    feature_calculators = get_feature_calculators(fps)
    feature_keys = ['Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg']
    feature_names = [
        f"{key}_{name}" for key in feature_keys 
        for name, _ in feature_calculators
    ]
    
    # 初始化输出
    n_windows = len(range(0, len(features), step_size_frames))
    features_array = np.zeros((n_windows, len(feature_names)))
    window_metadata = []
    
    for i, start_idx in enumerate(range(0, len(features), step_size_frames)):
        end_idx = min(start_idx + window_size_frames, len(features))
        
        # 记录元数据
        window_metadata.append(utils.create_window_metadata(
            min_start_time * fps + start_idx,
            min_start_time * fps + end_idx,
            fps
        ))
        
        # 计算特征
        window_data = features[start_idx:end_idx]
        feature_values = []
        for key in feature_keys:
            motion = extract_motion_components(window_data, key)
            for _, calculator in feature_calculators:
                feature_values.append(calculator(motion))
        
        features_array[i, :] = feature_values
    
    return features_array, feature_names, window_metadata

# ================== 使用示例 ==================
if __name__ == "__main__":
    # 示例参数
    prefix = '/data/Leo/mm/data/Newborn200/'
    input_json = prefix + "Body/50cm3.24kg_motion_features.json"
    label_file = prefix + "Label/50cm3.24kg.txt"
    window_size = 2.5  # 2.5秒窗口大小
    step_size = 2.5    # 1.5秒步长
    
    # 提取特征
    features, feature_names, meta = body_features(input_json, window_size, step_size, label_file)
    
    print(f"特征矩阵形状: {features.shape}")
    print(f"前5个特征名称: {feature_names[:5]}")
    print(f"前5个窗口元数据: {meta[:5]}")  # 打印前5个窗口的元数据
