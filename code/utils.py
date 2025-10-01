import os
import random
import numpy as np
import torch
from typing import Dict, Tuple
import json

def initialize_random_seed(seed: int) -> None:
    """Initialize random seeds for reproducibility across libraries."""
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

def prepare_feature_output_path(dataDir: str, feature_type: str) -> Tuple[str, str]:
    """Generate output directory path for specific feature type."""
    base_name = os.path.splitext(os.path.basename(dataDir))[0]
    parent_dir = os.path.dirname(dataDir)
    new_parent = os.path.join(os.path.dirname(parent_dir), feature_type)
    os.makedirs(new_parent, exist_ok=True)
    return base_name, new_parent

def get_label_file_path(dataDir: str) -> str:
    """Construct path for label file based on data directory."""
    base_name, new_parent = prepare_feature_output_path(dataDir, "Label")
    return os.path.join(new_parent, f"{base_name}.txt")

def get_scene_file_path(dataDir: str) -> str:
    """Construct path for scene file based on data directory."""
    base_name, new_parent = prepare_feature_output_path(dataDir, "SCENE")
    return os.path.join(new_parent, f"{base_name}.txt")

def get_motion_feature_file_path(dataDir: str) -> str:
    """Construct path for motion feature JSON file."""
    base_name, new_parent = prepare_feature_output_path(dataDir, "Body")
    return os.path.join(new_parent, f"{base_name}_motion_features.json")

def get_face_landmark_file_path(dataDir: str) -> str:
    """Construct path for facial landmark JSON file."""
    base_name, new_parent = prepare_feature_output_path(dataDir, "Face")
    return os.path.join(new_parent, f"{base_name}_face_landmarks.json")

def extract_subject_id(filepath: str) -> str:
    """Extract subject ID from file path (filename without extension)."""
    return os.path.splitext(os.path.basename(filepath))[0]
    
def generate_feature_filename(label_path: str) -> str:
    """Generate feature filename based on label file path."""
    basename = os.path.splitext(os.path.basename(label_path))[0]
    return f"{basename}_features.npz"

def load_and_validate_json(input_path: str) -> Dict:
    """Load JSON file and validate its existence."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    with open(input_path) as f:
        return json.load(f)
    
def get_valid_time_range(label_path: str) -> Tuple[float, float]:
    """Extract time range from label file (min start to max end time)."""
    min_start_time, max_end_time = float('inf'), 0.0
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    min_start_time = min(min_start_time, float(parts[0]))
                    max_end_time = max(max_end_time, float(parts[1]))
    return min_start_time, max_end_time

def crop_data_by_time(
    data: list, 
    fps: float, 
    min_start_time: float, 
    max_end_time: float
) -> list:
    """Crop data list to specified time range using frame rate."""
    start_frame = int(min_start_time * fps)
    end_frame = int(max_end_time * fps)
    return data[start_frame:end_frame]

def calculate_window_params(
    fps: float, 
    window_size_sec: float, 
    step_size_sec: float
) -> Tuple[int, int]:
    """Convert time-based window parameters to frame-based."""
    return (
        int(round(window_size_sec * fps)),
        int(round(step_size_sec * fps))
    )

def filter_outliers(values: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Replace outliers with NaNs using IQR method if outlier ratio < threshold."""
    if values.size < 10 or np.sum(~np.isnan(values)) < 10:
        return values
    
    valid_values = values[~np.isnan(values)]
    Q1, Q3 = np.nanpercentile(valid_values, [25, 75])
    IQR = Q3 - Q1
    upper_bound = Q3 + 3.0 * IQR
    lower_bound = Q1 - 3.0 * IQR
    
    outliers = (values > upper_bound) | (values < lower_bound)
    outlier_ratio = np.sum(outliers) / values.size

    if outlier_ratio < threshold:
        values[outliers] = np.nan

    return values