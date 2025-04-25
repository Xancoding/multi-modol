import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import os
import matplotlib as mpl

# 调整参数
mpl.rcParams['agg.path.chunksize'] = 10000  # 增加路径分块大小
mpl.rcParams['path.simplify_threshold'] = 1.0  # 增加路径简化阈值

def read_wav_file(file_path):
    # 读取WAV文件
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def ensure_directory_exists(directory):
    """
    检查文件夹是否存在，如果不存在则创建。
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_waveform(data, sample_rate, file, label_file, start_time=None, end_time=None, highlight_crying=True):
    """
    绘制音频波形图，并格式化横轴为时间轴。
    :param data: 音频数据
    :param sample_rate: 采样率
    :param file: 音频文件路径
    :param label_file: 标签文件路径
    :param start_time: 开始时间（秒），如果为 None，则绘制完整图像
    :param end_time: 结束时间（秒），如果为 None，则绘制完整图像
    :param highlight_crying: 是否将哭的部分的线变为红色，默认为 True
    """
    # 提取婴儿名字
    baby_name = file.split('\\')[1]
    directory = f"output/{baby_name}"
    ensure_directory_exists(directory)

    # 计算时间轴
    time = np.arange(0, len(data)) / sample_rate

    # 如果未指定时间段，则使用完整的时间范围
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = time[-1]

    # 找到给定时间段的索引
    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)

    # 提取给定时间段的数据
    segment_time = time[start_idx:end_idx]
    segment_data = data[start_idx:end_idx]

    # 绘制波形图
    plt.figure(figsize=(40, 16))  # 增大图像大小

    if highlight_crying:
        # 读取标签文件
        with open(label_file, 'r') as f:
            segments = [tuple(map(float, line.strip().split())) for line in f]

        # 分段绘制波形
        for seg_start, seg_end, label in segments:
            # 只绘制与给定时间段重叠的部分
            overlap_start = max(seg_start, start_time)
            overlap_end = min(seg_end, end_time)
            if overlap_start < overlap_end:
                overlap_start_idx = int(overlap_start * sample_rate) - start_idx
                overlap_end_idx = int(overlap_end * sample_rate) - start_idx
                overlap_time = segment_time[overlap_start_idx:overlap_end_idx]
                overlap_data = segment_data[overlap_start_idx:overlap_end_idx]
                if label == 1:  # 哭的部分
                    plt.plot(overlap_time, overlap_data, color='red', linewidth=1.5)  # 加粗红色线
                else:  # 非哭的部分
                    plt.plot(overlap_time, overlap_data, color='blue', linewidth=1.5)  # 加粗蓝色线
    else:
        # 如果不标记哭的部分，直接绘制整个波形
        plt.plot(segment_time, segment_data, color='blue', linewidth=1.5)  # 加粗蓝色线

    plt.title(f'Waveform', fontsize=52)
    plt.xlabel('Time [s]', fontsize=50)
    plt.ylabel('Amplitude', fontsize=50)

    # 格式化横轴
    xticks = np.arange(start_time, end_time + 1, 20)  # 每 5 秒一个刻度
    xtick_labels = [f"{int(tick // 60):02d}:{int(tick % 60):02d}" for tick in xticks]  # 转换为分钟:秒格式
    plt.xticks(xticks, xtick_labels, fontsize=50)
    plt.yticks(fontsize=50)

    # 保存图像
    plt.savefig(f"{directory}/audio_origin.png", dpi=300)  # 提高分辨率
    plt.close()  # 关闭图像，避免内存泄漏


if __name__ == "__main__":
    # 查找所有 WAV 文件
    files = glob.glob("infant/*/*.wav")
    for file in files:
        print(f"Processing audio: {file}")
        label_file = f"{file.split('_aduio.wav')[0]}_label.txt"
        # 读取 WAV 文件
        sample_rate, data = read_wav_file(file)

        # 定义时间段（如果不需要时间段，则设置为 None）
        start_time = 300  # 设置为 None 以绘制完整图像
        end_time = 500    # 设置为 None 以绘制完整图像

        # 绘制波形图
        plot_waveform(data, sample_rate, file, label_file, start_time, end_time, highlight_crying=True)