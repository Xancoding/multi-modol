import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import matplotlib.mlab as mlab
import json
from datetime import timedelta
from math import isnan

def read_wav(file_path):
    """Read WAV file and convert to mono"""
    rate, data = wavfile.read(file_path)
    return rate, data.mean(axis=1) if data.ndim == 2 else data

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

def plot_combined_spectrogram_and_features(audio_file, motion_file, face_file, output_file, start_time=None, end_time=None):
    """Plot spectrogram with motion intensity and mouth aspect ratio overlaid"""
    # Read audio data
    rate, data = read_wav(audio_file)
    total_duration = len(data) / rate

    # Process time range
    start_sec = time_str_to_seconds(start_time) if start_time else 0
    end_sec = time_str_to_seconds(end_time) if end_time else total_duration

    # Extract audio segment
    start_idx = int(start_sec * rate)
    end_idx = int(end_sec * rate)
    segment = data[start_idx:end_idx]

    # Generate spectrogram
    spec, freqs, t = mlab.specgram(segment, Fs=rate, NFFT=1024, noverlap=512)
    t = np.linspace(start_sec, end_sec, len(t))

    # Load feature data
    motion_times, motion_values = load_feature_data(motion_file, 'motion')
    mar_times, mar_values = load_feature_data(face_file, 'mar')

    # Filter feature data to time range
    motion_filter = (motion_times >= start_sec) & (motion_times <= end_sec)
    mar_filter = (mar_times >= start_sec) & (mar_times <= end_sec)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot spectrogram
    spec_plot = ax1.pcolormesh(t, freqs, 10*np.log10(spec), cmap='viridis', shading='auto')
    ax1.set_ylim(0, freqs[-1])
    ax1.set_yticks([])  # Remove y-axis ticks and labels for spectrogram

    # Remove colorbar for spectrogram
    # plt.colorbar(spec_plot, ax=ax1, label='Intensity [dB]')  # Comment out to remove energy annotation

    # Create secondary axis for motion intensity (left side)
    ax1_motion = ax1.twinx()
    ax1_motion.spines['left'].set_position(('outward', 0))
    ax1_motion.plot(
        motion_times[motion_filter],
        motion_values[motion_filter],
        color='tab:red',
        linewidth=2,
        label='Motion Intensity'
    )
    ax1_motion.set_ylabel('Motion Intensity', color='tab:red')
    ax1_motion.tick_params(axis='y', labelcolor='tab:red')
    ax1_motion.yaxis.set_label_position('left')
    ax1_motion.yaxis.set_ticks_position('left')

    # Create secondary axis for mouth aspect ratio (right side)
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('outward', 0))
    ax2.plot(
        mar_times[mar_filter],
        mar_values[mar_filter],
        color='tab:blue',
        linewidth=2,
        label='Mouth Aspect Ratio'
    )
    ax2.set_ylabel('Mouth Aspect Ratio', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.set_ticks_position('right')

    # Set time axis
    ax1.set_xlim(start_sec, end_sec)
    ax1.set_xlabel('Time')

    # Set time ticks
    time_ticks = np.linspace(start_sec, end_sec, 5)
    ax1.set_xticks(time_ticks)
    ax1.set_xticklabels([seconds_to_time_str(t) for t in time_ticks], rotation=45)

    # Add legend for feature plots
    lines_motion, labels_motion = ax1_motion.get_legend_handles_labels()
    lines_mar, labels_mar = ax2.get_legend_handles_labels()
    ax1.legend(lines_motion + lines_mar, labels_motion + labels_mar, loc='upper right')

    # Add minimal grid
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Add title
    # plt.title('Audio Spectrogram with Motion Intensity and Mouth Aspect Ratio')

    # Save plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined plot saved to: {output_file}")
    print(f"Time segment: {seconds_to_time_str(start_sec)} - {seconds_to_time_str(end_sec)}")
    print(f"Total duration: {seconds_to_time_str(total_duration)}")
    print(f"Analysis duration: {seconds_to_time_str(end_sec - start_sec)}")

if __name__ == "__main__":
    prefix = "/data/Leo/mm/data/NanfangHospital/"
    infant = "lxm-baby-m_2025-07-29-14-55-30"
    audio_file = f"{prefix}data/{infant}.wav"
    motion_file = f"{prefix}Body/{infant}_motion_features.json"
    face_file = f"{prefix}Face/{infant}_face_landmarks.json"
    
    plot_combined_spectrogram_and_features(
        audio_file,
        motion_file,
        face_file,
        "synplot.png",
        start_time="00:00:45.000",
        end_time="00:00:47.500"
    )