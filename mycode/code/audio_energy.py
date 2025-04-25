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

def calculate_energy(data, window_size, sample_rate, start_time=None, end_time=None):
    """
    计算音频信号的能量随时间的变化。
    :param data: 音频数据
    :param window_size: 窗口大小（采样点数）
    :param sample_rate: 采样率
    :param start_time: 开始时间（秒），如果为 None，则计算完整时间段
    :param end_time: 结束时间（秒），如果为 None，则计算完整时间段
    :return: 时间轴和能量值
    """
    # 如果未指定时间段，则使用完整的时间范围
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = len(data) / sample_rate

    # 找到给定时间段的索引
    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)

    # 提取给定时间段的数据
    segment_data = data[start_idx:end_idx]

    # 计算能量
    energy = []
    for i in range(0, len(segment_data), window_size):
        window = segment_data[i:i + window_size]
        if len(window) > 0:
            energy.append(np.sum(np.square(window)))

    # 计算时间轴
    time = np.arange(len(energy)) * (window_size / sample_rate) + start_time
    return time, energy

def plot_energy(data, sample_rate, file, label_file, window_size=16000, start_time=None, end_time=None):
    """
    绘制音频能量随时间的变化图，并标记标签为1的部分为红色。
    :param data: 音频数据
    :param sample_rate: 采样率
    :param file: 音频文件路径
    :param label_file: 标签文件路径
    :param window_size: 窗口大小（采样点数）
    :param start_time: 开始时间（秒），如果为 None，则绘制完整图像
    :param end_time: 结束时间（秒），如果为 None，则绘制完整图像
    """
    # 提取婴儿名字
    baby_name = file.split('\\')[1]
    directory = f"output/{baby_name}"
    ensure_directory_exists(directory)

    # 计算能量随时间的变化
    time, energy = calculate_energy(data, window_size, sample_rate, start_time, end_time)

    # 将 energy 转换为 NumPy 数组
    energy = np.array(energy)

    # 绘制能量图
    plt.figure(figsize=(40, 16))  # 增大图像大小

    # 读取标签文件
    with open(label_file, 'r') as f:
        segments = [tuple(map(float, line.strip().split())) for line in f]

    # 分段绘制能量图
    for seg_start, seg_end, label in segments:
        # 只绘制与给定时间段重叠的部分
        overlap_start = max(seg_start, start_time if start_time is not None else 0)
        overlap_end = min(seg_end, end_time if end_time is not None else time[-1])
        if overlap_start < overlap_end:
            overlap_indices = (time >= overlap_start) & (time <= overlap_end)
            overlap_time = time[overlap_indices]
            overlap_energy = energy[overlap_indices]
            if label == 1:  # 标签为1的部分
                plt.plot(overlap_time, overlap_energy, color='red', linewidth=1.5)  # 红色线
            else:  # 其他部分
                plt.plot(overlap_time, overlap_energy, color='green', linewidth=1.5)  # 绿色线

    plt.title('Audio Energy', fontsize=52)
    plt.xlabel('Time [s]', fontsize=50)
    plt.ylabel('Energy', fontsize=50)

    # 格式化横轴
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = time[-1]
    xticks = np.arange(start_time, end_time + 1, 20)  # 每 20 秒一个刻度
    xtick_labels = [f"{int(tick // 60):02d}:{int(tick % 60):02d}" for tick in xticks]  # 转换为分钟:秒格式
    plt.xticks(xticks, xtick_labels, fontsize=50)
    plt.yticks(fontsize=50)

    # 保存图像
    plt.savefig(f"{directory}/audio_energy.png", dpi=300)  # 提高分辨率
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

        # start_time = 100  # 设置为 None 以绘制完整图像
        # end_time = 300  # 设置为 None 以绘制完整图像

        # 绘制能量图
        plot_energy(data, sample_rate, file, label_file, start_time=start_time, end_time=end_time)