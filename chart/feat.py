import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import timedelta
from math import isnan

def seconds_to_time_str(total_seconds):
    """Convert seconds to HH:MM:SS.SSS format"""
    time_delta = timedelta(seconds=total_seconds)
    time_part, milli_part = str(time_delta).split('.') if '.' in str(time_delta) else (str(time_delta), "000")
    return f"{time_part}.{milli_part[:3]}"

def time_str_to_seconds(time_string):
    """Convert HH:MM:SS.SSS to total seconds"""
    hours, minutes, seconds = time_string.split(':')
    sec_part, milli_part = seconds.split('.') if '.' in seconds else (seconds, "0")
    return float(hours)*3600 + float(minutes)*60 + float(sec_part) + float(milli_part)/1000

def get_peak_motion(motion_records):
    """Get highest confidence motion intensity"""
    return max(motion_records, key=lambda x: x[4])[2] if motion_records else None

def compute_mouth_aspect_ratio(facial_landmarks):
    """Calculate Mouth Aspect Ratio"""
    if len(facial_landmarks) < 68: return 0.0
    try:
        points = np.array(facial_landmarks)
        mouth_width = np.linalg.norm(points[54]-points[48])
        lip_height = (np.linalg.norm(points[58]-points[50]) + np.linalg.norm(points[56]-points[52]))/2
        return round(lip_height/mouth_width, 4) if mouth_width > 1e-6 else 0.0
    except:
        return 0.0

def load_feature_data(file_path, data_type):
    """Load and process feature data file"""
    with open(file_path) as file:
        json_data = json.load(file)
    
    frame_rate = json_data['video_info']['fps']
    data_records = json_data['features' if data_type == 'motion' else 'frames']
    frame_id_key = 'Frame' if data_type == 'motion' else 'frame_number'
    
    values, timestamps = [], []
    for record in data_records:
        if data_type == 'motion' and 'WholeBody' in record:
            if intensity := get_peak_motion(record['WholeBody']):
                values.append(intensity)
                timestamps.append(record[frame_id_key]/frame_rate)
        elif data_type == 'mar' and 'landmarks' in record:
            values.append(compute_mouth_aspect_ratio(record['landmarks']))
            timestamps.append(record[frame_id_key]/frame_rate)
    
    return np.array(timestamps), np.array(values)

def visualize_features(body_data_path, face_data_path, output_file=None, start_time=None, end_time=None):
    """Simplified visualization of motion intensity and mouth aspect ratio"""
    # Load datasets
    motion_times, motion_values = load_feature_data(body_data_path, 'motion')
    mar_times, mar_values = load_feature_data(face_data_path, 'mar')
    
    # Determine analysis window
    start_sec = time_str_to_seconds(start_time) if start_time else max(motion_times[0], mar_times[0])
    end_sec = time_str_to_seconds(end_time) if end_time else min(motion_times[-1], mar_times[-1])
    
    # Filter data to time range
    motion_filter = (motion_times >= start_sec) & (motion_times <= end_sec)
    mar_filter = (mar_times >= start_sec) & (mar_times <= end_sec)
    
    # Create figure with clean styling
    fig, primary_axis = plt.subplots(figsize=(12, 6))
    
    # Plot motion data as solid line
    motion_line = primary_axis.plot(
        motion_times[motion_filter], 
        motion_values[motion_filter],
        color='tab:red',
        linewidth=2,
        label='Motion Intensity'
    )
    primary_axis.set_ylabel('Motion Intensity', color='tab:red')
    
    # Add MAR axis with solid line
    secondary_axis = primary_axis.twinx()
    mar_line = secondary_axis.plot(
        mar_times[mar_filter], 
        mar_values[mar_filter],
        color='tab:blue',
        linewidth=2,
        label='Mouth Aspect Ratio'
    )
    secondary_axis.set_ylabel('Mouth Aspect Ratio', color='tab:blue')
    
    # Set time axis
    primary_axis.set_xlim(start_sec, end_sec)
    primary_axis.set_xlabel('Time (HH:MM:SS.SSS)')
    
    # Set time ticks
    time_ticks = np.linspace(start_sec, end_sec, 5)
    primary_axis.set_xticks(time_ticks)
    primary_axis.set_xticklabels([seconds_to_time_str(t) for t in time_ticks], rotation=45)
    
    # Add minimal grid
    primary_axis.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    prefix = "/data/Leo/mm/data/Newborn200/"
    infant = "46cm2.4kg"
    motion_file = f"{prefix}Body/{infant}_motion_features.json"
    face_file = f"{prefix}Face/{infant}_face_landmarks.json"
    
    visualize_features(
        motion_file,
        face_file,
        "feat.png",
        start_time="00:00:10.000",
        end_time="00:00:12.500"
    )