import numpy as np
from typing import List, Dict, Tuple, Callable
from math import isnan
import utils

# Constants for facial landmarks indices
MOUTH_POINTS = (48, 54, 56, 58, 50, 52)
LEFT_EYE_POINTS = range(36, 42)
RIGHT_EYE_POINTS = range(42, 48)
_face_stats = {'total': 0, 'errors': 0}

def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_mar(landmarks: List[List[float]]) -> float:
    """Calculate Mouth Aspect Ratio (MAR)"""
    try:
        if len(landmarks) < 68:
            return np.nan
        
        p48, p54, p56, p58, p50, p52 = [landmarks[i] for i in MOUTH_POINTS]
        
        mouth_width = calculate_distance(p54, p48)
        if mouth_width < 1e-6:
            return np.nan
            
        left_lip = calculate_distance(p58, p50)
        right_lip = calculate_distance(p56, p52)
        
        mar = (left_lip + right_lip) / (2 * mouth_width)
        return round(float(mar), 4) if not isnan(mar) else np.nan
    except Exception:
        return np.nan

def calculate_eye_aspect_ratio(eye_points: List[List[float]]) -> float:
    """Calculate Eye Aspect Ratio for single eye"""
    A = calculate_distance(eye_points[1], eye_points[5])
    B = calculate_distance(eye_points[2], eye_points[4])
    C = calculate_distance(eye_points[0], eye_points[3])
    return (A + B) / (2 * C) if C > 1e-6 else np.nan

def calculate_ear(landmarks: List[List[float]]) -> Tuple[float, float]:
    """Calculate Eye Aspect Ratio (EAR) for both eyes"""
    try:
        if len(landmarks) < 68:
            return (np.nan, np.nan)
        
        left_eye = [landmarks[i] for i in LEFT_EYE_POINTS]
        right_eye = [landmarks[i] for i in RIGHT_EYE_POINTS]
        
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        
        return (
            round(float(left_ear), 4) if not isnan(left_ear) else np.nan,
            round(float(right_ear), 4) if not isnan(right_ear) else np.nan
        )
    except Exception:
        return (np.nan, np.nan)

def create_feature_calculator(func: Callable, feature_key: str) -> Callable:
    """Factory function to create feature calculators"""
    return lambda data: func(data[feature_key])

def get_facial_feature_calculators() -> List[Tuple[str, Callable]]:
    """Generate facial feature calculation functions"""
    FEATURES = {
        'mar': 'mars',
        'left_ear': 'left_ears',
        'right_ear': 'right_ears'
    }
    
    STATS = {
        'mean': lambda arr: 0.0 if np.all(np.isnan(arr)) else np.nanmean(arr),
        'median': lambda arr: 0.0 if np.all(np.isnan(arr)) else np.nanmedian(arr),
        'std': lambda arr: 0.0 if np.all(np.isnan(arr)) else np.nanstd(arr),
        'min': lambda arr: 0.0 if np.all(np.isnan(arr)) else np.nanmin(arr),
        'max': lambda arr: 0.0 if np.all(np.isnan(arr)) else np.nanmax(arr)
    }

    calculators = []
    for feature_name, data_key in FEATURES.items():
        for stat_name, stat_func in STATS.items():
            name = f"{feature_name}_{stat_name}"
            calculator = create_feature_calculator(stat_func, data_key)
            calculators.append((name, calculator))
    
    return calculators

def extract_window_features(window_frames: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract and process features for a time window"""
    landmarks_list = [frame.get('landmarks', []) for frame in window_frames]
    _face_stats['total'] += len(landmarks_list)
    
    raw_feats = {
        'mars': np.array([calculate_mar(lm) for lm in landmarks_list]),
        'left_ears': np.array([ear[0] for ear in 
                             [calculate_ear(lm) for lm in landmarks_list]]),
        'right_ears': np.array([ear[1] for ear in 
                              [calculate_ear(lm) for lm in landmarks_list]])
    }

    for feat in raw_feats.values():
        _face_stats['errors'] += np.sum(np.isnan(feat))
    feats = {}
    for k, v in raw_feats.items():
        cleaned = utils.filter_outliers(v.copy())
        feats[k] = cleaned 
        _face_stats['errors'] += np.sum(np.isnan(cleaned)) - np.sum(np.isnan(v))
    return feats

    # # Apply outlier filtering to each feature
    # return {k: utils.filter_outliers(v) for k, v in raw_feats.items()}    

def facial_features(
    input_path: str,
    window_size_sec: float,
    step_size_sec: float,
    label_path: str
) -> Tuple[np.ndarray, List[str]]:
    """Main pipeline for facial feature extraction"""
    try:
        data = utils.load_and_validate_json(input_path)
        fps = data['video_info']['fps']
        frames = utils.crop_data_by_time(
            data.get('frames', []),
            fps,
            *utils.get_valid_time_range(label_path)
        )
        
        window_size = int(fps * window_size_sec)
        step_size = int(fps * step_size_sec)
        n_frames = len(frames)
        
        # Generate valid window indices
        window_starts = range(0, n_frames - window_size + 1, step_size)
        
        # Get feature calculators
        calculators = get_facial_feature_calculators()
        feature_names = [name for name, _ in calculators]
        
        # Process each window
        features = np.array([
            [calc(extract_window_features(frames[start:start+window_size]))
             for _, calc in calculators]
            for start in window_starts
        ])

        
        return features, feature_names
        
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        n_features = len(get_facial_feature_calculators())
        return np.zeros((0, n_features)), []

