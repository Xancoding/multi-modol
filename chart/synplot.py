import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import matplotlib.mlab as mlab
import json
import os
import pandas as pd
from PIL import Image

# --- 全局常量 (用于统一绘图范围) ---
# 声谱图 VMIN/VMAX，用于统一所有场景的颜色映射范围
GLOBAL_SPECTROGRAM_VMIN = None
GLOBAL_SPECTROGRAM_VMAX = None
# 运动强度 (MI) Y 轴范围 (初始值，将被预计算覆盖)
GLOBAL_MI_YMIN = 0.0
GLOBAL_MI_YMAX = 1.5
# 口部纵横比 (MAR) Y 轴范围 (初始值，将被预计算覆盖，最大值不超过 1.0)
GLOBAL_MAR_YMIN = 0.0
# GLOBAL_MAR_YMAX = 1.0
# 平滑窗口大小
GLOBAL_WINDOW_SIZE = 1  
# 字体大小
LABELSIZE = 20 # 太大
# LABELSIZE = 12
LINE_WIDTH = 4


# --- 工具函数 ---

def read_wav(file_path):
    """读取 WAV 文件，并将立体声转换为单声道 (取平均值)。"""
    try:
        rate, data = wavfile.read(file_path)
        # 检查声道数，如果是立体声 (ndim=2)，则取平均值转换为单声道
        return rate, data.mean(axis=1) if data.ndim == 2 else data
    except Exception as e:
        print(f"Error reading WAV {file_path}: {e}")
        raise

def time_str_to_seconds(time_string):
    """将时间字符串转换为浮点数秒数。"""
    return float(time_string)

def smooth_data(values, window_size=5):
    """使用滚动平均对数据进行平滑处理。"""
    if values.size < window_size or np.sum(~np.isnan(values)) < 2:
        return values
    
    # 将 numpy 数组转换为 pandas Series，进行滚动平均，然后再转回 numpy
    # min_periods=1 确保了序列开头和结尾能被平滑处理
    # center=True 使平滑窗口以当前点为中心
    return pd.Series(values).rolling(
        window=window_size, 
        min_periods=1, 
        center=True
    ).mean().to_numpy()

def get_peak_motion(motion_records):
    """从 WholeBody/Face 记录中提取置信度最大的运动强度值。
    记录格式: (x, y, intensity, size, confidence)"""
    # 找到 confidence (索引 4) 最大的记录，返回其 intensity (索引 2)
    return max(motion_records, key=lambda x: x[4])[2] if motion_records else None

def compute_mouth_aspect_ratio(facial_landmarks):
    """计算口部纵横比 (MAR)。使用 68 点标注。"""
    if not facial_landmarks or len(facial_landmarks) < 68:
        return None
    try:
        pts = np.array(facial_landmarks)
        # 嘴唇点位 (48-67)：水平宽度使用 54(右) - 48(左)
        width = np.linalg.norm(pts[54] - pts[48])
        # 垂直高度使用 (58(下内) - 50(上内)) 和 (56(下中) - 52(上中)) 的平均
        height = (np.linalg.norm(pts[58] - pts[50]) + np.linalg.norm(pts[56] - pts[52])) / 2
        # 防止除以零
        return height / width if width > 1e-6 else None
    except Exception:
        # 捕获可能因数据格式问题导致的错误
        return None

def load_feature_data(file_path, data_type, start_time=0, end_time=float('inf'), motion_key='WholeBody'):
    """
    加载特征数据 (运动强度 MI 或 MAR)，并根据时间窗口过滤。
    
    参数:
        file_path (str): JSON 文件路径。
        data_type (str): 'motion' 或 'mar'。
        start_time (float): 起始时间 (秒)。
        end_time (float): 结束时间 (秒)。
        motion_key (str): 加载 'motion' 数据时，指定 'WholeBody' 或 'Face'。
        
    返回: 
        (numpy.array, numpy.array): (timestamps, values)
    """
    try:
        with open(file_path) as f:
            json_data = json.load(f)
    except Exception:
        # 文件读取或解析失败时返回空数组
        return np.array([]), np.array([])

    fps = json_data.get('video_info', {}).get('fps', 30)
    resolution = json_data.get('video_info', {}).get('resolution', [720.0, 1280.0])
    # 特征记录的键名在 'motion' 和 'mar' 文件中不同
    records = json_data.get('features' if data_type == 'motion' else 'frames', [])
    frame_key = 'Frame' if data_type == 'motion' else 'frame_number'

    timestamps, values = [], []
    
    for rec in records:
        # 计算当前帧的时间
        frame_time = rec.get(frame_key, 0) / fps
        if start_time <= frame_time <= end_time:
            if data_type == 'motion' and motion_key in rec:
                motion_records = rec.get(motion_key) 
                intensity = get_peak_motion(motion_records) if motion_records else None
                    
                values.append(intensity if intensity is not None else np.nan)
                timestamps.append(frame_time)  
            elif data_type == 'mar' and 'landmarks' in rec:
                mar = compute_mouth_aspect_ratio(rec['landmarks'])
                values.append(mar if mar is not None else np.nan)
                timestamps.append(frame_time)

    return np.array(timestamps), np.array(values)

# --- 绘图和拼接函数 ---

def stitch_plots_horizontally(file_list, output_path="stitched_scenes_horizontal.png"):
    """将给定的 PNG 图像列表水平拼接成一张大图。"""
    images = []
    for file_path in file_list:
        try:
            images.append(Image.open(file_path))
        except Exception as e:
            print(f"Error opening image {file_path}: {e}")

    if not images:
        print("Error: No images were loaded successfully.")
        return

    # 计算总宽度和最大高度
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # 创建新的空白图片
    stitched_image = Image.new('RGB', (total_width, max_height))
    
    # 逐个粘贴图片
    current_x = 0
    for img in images:
        stitched_image.paste(img, (current_x, 0))
        current_x += img.width

    stitched_image.save(output_path)
    print(f"\nSuccessfully stitched {len(images)} plots horizontally to {output_path}")

def filter_outliers(values):
    """基于 IQR (K=3.0) 过滤异常值，将其替换为 NaN。"""
    # 至少需要 10 个有效值才能计算 IQR
    if values.size < 10 or np.sum(~np.isnan(values)) < 10:
        return values
    
    valid_values = values[~np.isnan(values)]
    Q1, Q3 = np.nanpercentile(valid_values, [25, 75])
    IQR = Q3 - Q1
    # 上下界使用 3.0 * IQR (更加宽松的异常值定义)
    upper_bound = Q3 + 3.0 * IQR
    lower_bound = Q1 - 3.0 * IQR
    
    # 将超出上下界的点标记为 NaN
    values[np.where(values > upper_bound)[0]] = np.nan
    values[np.where(values < lower_bound)[0]] = np.nan
    return values

def plot_combined_spectrogram_and_features(audio_file, motion_file, face_file, output_file, start_time=None, end_time=None, motion_key='WholeBody'):
    """
    绘制声谱图 (底图)、运动强度 (左 Y 轴) 和 MAR (右 Y 轴) 的组合图。
    声谱图和特征图的 Y 轴范围将使用全局常量统一。
    """
    global GLOBAL_SPECTROGRAM_VMIN, GLOBAL_SPECTROGRAM_VMAX
    global GLOBAL_MI_YMIN, GLOBAL_MI_YMAX, GLOBAL_MAR_YMIN, GLOBAL_MAR_YMAX

    try:
        rate, data = read_wav(audio_file)
    except Exception:
        print(f"Skipping plot due to audio error: {audio_file}")
        return

    total_duration = len(data) / rate
    start_sec = time_str_to_seconds(start_time) if start_time else 0
    end_sec = time_str_to_seconds(end_time) if end_time else total_duration
    # 截取音频片段
    segment = data[int(start_sec * rate):int(end_sec * rate)]

    # 声谱图计算
    # 使用 mlab.specgram 计算，返回频谱、频率、时间轴
    spec, freqs, t_spec = mlab.specgram(segment, Fs=rate, NFFT=1024, noverlap=512)
    t = np.linspace(start_sec, end_sec, len(t_spec)) # 调整时间轴与场景时间匹配
    spec_db = 10 * np.log10(spec) # 转换为 dB

    # 特征加载
    # motion_key 传入 load_feature_data 以加载正确的 MI ('WholeBody' 或 'Face')
    motion_times, motion_values = load_feature_data(motion_file, 'motion', start_sec, end_sec, motion_key)
    # MAR 加载不依赖 motion_key
    mar_times, mar_values = load_feature_data(face_file, 'mar', start_sec, end_sec)
    
    # 应用运动和 MAR 异常值过滤
    motion_values = filter_outliers(motion_values)
    mar_values = filter_outliers(mar_values)

    motion_values = smooth_data(motion_values, GLOBAL_WINDOW_SIZE) 
    mar_values = smooth_data(mar_values, GLOBAL_WINDOW_SIZE)

    # 打印有效数据点统计信息
    mar_valid = np.sum(~np.isnan(mar_values)) 
    motion_valid = np.sum(~np.isnan(motion_values))
    print(f"Motion valid/total: {motion_valid}/{len(motion_values)} ({100*motion_valid/(len(motion_values) or 1):.1f}%)")
    print(f"MAR valid/total: {mar_valid}/{len(mar_values)} ({100*mar_valid/(len(mar_values) or 1):.1f}%)")

    # --- 绘图部分 ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 1. 声谱图 (底层图)
    # 使用预计算的全局范围 GLOBAL_SPECTROGRAM_VMIN/VMAX 统一颜色映射
    if GLOBAL_SPECTROGRAM_VMIN is not None:
        ax1.pcolormesh(t, freqs, spec_db, cmap='viridis', shading='auto', 
                       vmin=GLOBAL_SPECTROGRAM_VMIN, vmax=GLOBAL_SPECTROGRAM_VMAX)
    else:
        ax1.pcolormesh(t, freqs, spec_db, cmap='viridis', shading='auto')
        
    ax1.set_ylim(0, freqs[-1]) # 频率范围
    ax1.set_xlim(start_sec, end_sec) # 时间范围
    # ax1.set_xlabel('Time', fontsize=LABELSIZE + 2)
    ax1.set_yticks([]) # 隐藏声谱图的 Y 轴刻度
    ax1.set_xticks([]) # 隐藏 X 轴刻度，因为特征图的 X 轴已经足够
    ax1.grid(True, linestyle=':', alpha=0.5)

    
    # 2. 运动强度 (MI) - 左 Y 轴
    ax_motion = ax1.twinx() # 创建共享 X 轴的副坐标轴
    # ax_motion.plot(motion_times, motion_values, color='tab:red', linewidth=2, label=f'Somatic Motion Intensity (SMI)')
    ax_motion.plot(motion_times, motion_values, color='#FF4136', linewidth=LINE_WIDTH)    
    # 使用全局 MI 范围统一 Y 轴
    ax_motion.set_ylim(0.0, GLOBAL_MI_YMAX)
    
    ax_motion.tick_params(axis='y', labelcolor='#FF4136', labelsize=LABELSIZE)
    ax_motion.yaxis.set_label_position('left')
    ax_motion.yaxis.set_ticks_position('left')
    # ax_motion.set_yticklabels([])
    
    # ax_motion.set_ylabel(
    #     'SMI', 
    #     color='tab:red', 
    #     fontsize=LABELSIZE + 2, # 标签可以比刻度略大
    #     rotation=90
    # ) 

    lines, labels = ax_motion.get_legend_handles_labels()

    # 3. MAR - 右 Y 轴
    if mar_valid > 0:
        ax_mar = ax1.twinx() # 再次创建共享 X 轴的副坐标轴
        # ax_mar.plot(mar_times, mar_values, color='darkorange', linewidth=2, label='Mouth Aspect Ratio (MAR)')
        ax_mar.plot(mar_times, mar_values, color='#FFA500', linewidth=LINE_WIDTH)
        
        # 使用全局 MAR 范围统一 Y 轴
        ax_mar.set_ylim(0.0, GLOBAL_MAR_YMAX) 

        ax_mar.tick_params(axis='y', labelcolor='#FFA500', labelsize=LABELSIZE)
        ax_mar.yaxis.set_label_position('right')
        ax_mar.yaxis.set_ticks_position('right')
        # ax_mar.set_yticklabels([])

        # ax_mar.set_ylabel(
        #     'MAR', 
        #     color='darkorange', 
        #     fontsize=LABELSIZE + 2, # 标签可以比刻度略大
        #     rotation=270, # 将右侧标签旋转 270 度，使其从上往下阅读
        #     labelpad=15 # 增加标签和刻度之间的距离
        # ) 


        # 合并图例
        l2, lab2 = ax_mar.get_legend_handles_labels()
        lines.extend(l2)
        labels.extend(lab2)

    # 4. 图例和保存
    # ax1.legend(lines, labels, loc='upper right', fontsize=LABELSIZE + 2)
    plt.tight_layout() # 调整布局以适应标签
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def pre_calculate_spectrogram_range(scenes_list, prefix):
    """预计算所有场景声谱图的全局 VMIN 和 VMAX，用于统一颜色范围。"""
    global GLOBAL_SPECTROGRAM_VMIN, GLOBAL_SPECTROGRAM_VMAX
    
    all_db_values = []
    
    # 遍历所有场景，计算每个片段的声谱图 dB 极值
    for infant, start, end, *_ in scenes_list: 
        audio = os.path.join(prefix, "data", f"{infant}.wav")
        try:
            rate, data = read_wav(audio)
            
            start_sec = time_str_to_seconds(start)
            end_sec = time_str_to_seconds(end)
            segment = data[int(start_sec * rate):int(end_sec * rate)]

            # 计算声谱图 dB 值
            spec, _, _ = mlab.specgram(segment, Fs=rate, NFFT=1024, noverlap=512)
            spec_db = 10 * np.log10(spec)
            
            all_db_values.append(np.nanmin(spec_db))
            all_db_values.append(np.nanmax(spec_db))
            
        except Exception as e:
            print(f"Warning: Spectrogram pre-calculation failed for {infant} ({start}-{end}): {e}")
            continue

    if all_db_values:
        # 设定全局最小值和最大值，并添加 1.0 的容差/缓冲
        GLOBAL_SPECTROGRAM_VMIN = np.min(all_db_values) - 1.0 
        GLOBAL_SPECTROGRAM_VMAX = np.max(all_db_values) + 1.0
        print(f"Global Spectrogram Range set: VMIN={GLOBAL_SPECTROGRAM_VMIN:.2f}, VMAX={GLOBAL_SPECTROGRAM_VMAX:.2f}")

def pre_calculate_feature_ranges(scenes_list, prefix):
    """预计算所有场景的运动强度 (MI) 和 MAR 的全局 Y 轴范围。"""
    global GLOBAL_MI_YMIN, GLOBAL_MI_YMAX, GLOBAL_MAR_YMIN, GLOBAL_MAR_YMAX
    
    all_mi_values = []
    all_mar_values = []

    # 遍历所有场景，加载并过滤 MI 和 MAR 数据
    for infant, start, end, motion_key in scenes_list:
        motion_file = os.path.join(prefix, "Body", f"{infant}_motion_features.json")
        face_file = os.path.join(prefix, "Face", f"{infant}_face_landmarks.json")
        start_sec = time_str_to_seconds(start)
        end_sec = time_str_to_seconds(end)
        
        # 1. 加载运动数据
        _, motion_values = load_feature_data(motion_file, 'motion', start_sec, end_sec, motion_key)
        motion_values = filter_outliers(motion_values) # 应用异常值过滤
        motion_values = smooth_data(motion_values, GLOBAL_WINDOW_SIZE)
        all_mi_values.extend(motion_values[~np.isnan(motion_values)]) # 只收集非 NaN 值
        
        # 2. 加载 MAR 数据
        _, mar_values = load_feature_data(face_file, 'mar', start_sec, end_sec)
        mar_values = filter_outliers(mar_values) # 应用异常值过滤
        mar_values = smooth_data(mar_values, GLOBAL_WINDOW_SIZE)
        all_mar_values.extend(mar_values[~np.isnan(mar_values)])

    # --- 确定 MI 范围 ---
    if all_mi_values:
        mi_min = np.min(all_mi_values)
        mi_max = np.max(all_mi_values)
        mi_range = mi_max - mi_min
        # 添加 10% 的缓冲
        buffer = mi_range * 0.1 or 0.1
        GLOBAL_MI_YMIN = max(0.0, mi_min - buffer) # MI 最小值不低于 0.0
        GLOBAL_MI_YMAX = mi_max + buffer
    print(f"Global Motion Intensity Range set: YMIN={GLOBAL_MI_YMIN:.3f}, YMAX={GLOBAL_MI_YMAX:.3f}")

    # --- 确定 MAR 范围 ---
    if all_mar_values:
        mar_min = np.min(all_mar_values)
        mar_max = np.max(all_mar_values)
        mar_range = mar_max - mar_min
        # 添加 10% 的缓冲
        buffer = mar_range * 0.1 or 0.1
        GLOBAL_MAR_YMIN = max(0.0, mar_min - buffer)
        # MAR 范围特殊处理：上限不超过 1.0 (合理的 MAR 物理限制)
        GLOBAL_MAR_YMAX = min(1.0, mar_max + buffer) 
    print(f"Global MAR Range set: YMIN={GLOBAL_MAR_YMIN:.3f}, YMAX={GLOBAL_MAR_YMAX:.3f}")


if __name__ == "__main__":
    
    prefix = "/data/Leo/mm/data/NICU50" # 数据文件根路径


    # Face、Left-arm、Right-arm、Left-leg、Right-leg
    part = "Face"
    scenes = [
        ("ydw-baby-f_2025-07-29-13-39-36", "45.000", "47.500", part),
        ("zm-baby-m_2025-07-23-19-24-10", "545.000", "547.50", part),
        ("ysqd-f_2025-07-29-15-32-40", "301.500", "304.000", part),
        ("lxm-baby-m_2025-07-29-14-19-07", "4.000", "6.500", part),
        ("lxm-baby-m_2025-07-29-14-55-30", "170.000", "172.500", part),
        ("zm-baby-m_2025-07-23-19-24-10", "9.000", "11.500", part),
        ("lxm-baby-m_2025-07-29-14-55-30", "10.000", "12.500", part),
        ("zm-baby-m_2025-07-23-19-24-10", "5.000", "7.500", part),
    ]
    
    # **第一步：预计算所有全局范围**
    # 确保所有场景的声谱图颜色和特征 Y 轴范围统一
    pre_calculate_spectrogram_range(scenes, prefix)
    pre_calculate_feature_ranges(scenes, prefix)

    generated_files = [] 
    
    # **第二步：生成绘图文件**
    # 遍历场景列表，生成组合图并保存为 PNG
    for idx, (infant, start, end, motion_key) in enumerate(scenes):
        # 构造输入文件路径和输出文件名
        audio = os.path.join(prefix, "data", f"{infant}.wav")
        motion = os.path.join(prefix, "Body", f"{infant}_motion_features.json")
        face = os.path.join(prefix, "Face", f"{infant}_face_landmarks.json")
        out = f"scene{idx+1}.png"
        generated_files.append(out)
        
        print(f"\nProcessing scene {idx+1} (MI Type: {motion_key}): {infant} ({start}-{end})")
        # 调用绘图函数，传入 motion_key 确保加载正确的运动强度
        plot_combined_spectrogram_and_features(audio, motion, face, out, start, end, motion_key) 

    # **第三步：拼接图片**
    # 将所有生成的 PNG 文件水平拼接成一张最终的图片
    final_output_file = "combined.png"
    stitch_plots_horizontally(generated_files, final_output_file)