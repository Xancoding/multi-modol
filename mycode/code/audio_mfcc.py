import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import os
import matplotlib as mpl
import librosa

# 调整参数
mpl.rcParams['agg.path.chunksize'] = 10000  # 增加路径分块大小
mpl.rcParams['path.simplify_threshold'] = 1.0  # 增加路径简化阈值

def read_wav_file(file_path):
    """
    读取WAV文件。
    :param file_path: WAV文件路径
    :return: 采样率和音频数据
    """
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def ensure_directory_exists(directory):
    """
    检查文件夹是否存在，如果不存在则创建。
    :param directory: 文件夹路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_mfcc(data, sample_rate, start_time=None, end_time=None, n_mfcc=13):
    """
    计算音频信号的MFCC特征随时间的变化。
    :param data: 音频数据
    :param sample_rate: 采样率
    :param start_time: 开始时间（秒），如果为 None，则计算完整时间段
    :param end_time: 结束时间（秒），如果为 None，则计算完整时间段
    :param n_mfcc: 提取的MFCC系数数量
    :return: 时间轴和MFCC特征
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

    # 将数据转换为浮点型，并归一化到 [-1.0, 1.0] 范围
    if segment_data.dtype == np.int16:
        segment_data = segment_data.astype(np.float32) / 32768.0
    elif segment_data.dtype == np.int32:
        segment_data = segment_data.astype(np.float32) / 2147483648.0
    elif segment_data.dtype == np.float32:
        pass  # 已经是浮点型，无需处理
    else:
        raise ValueError("Unsupported audio data type")

    # 计算MFCC特征
    mfccs = librosa.feature.mfcc(y=segment_data, sr=sample_rate, n_mfcc=n_mfcc)

    # 计算时间轴
    time = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sample_rate) + start_time
    return time, mfccs

def plot_mfcc(data, sample_rate, file, label_file, start_time=None, end_time=None):
    """
    绘制MFCC特征随时间的变化图，并标记标签为1的部分为红色。
    :param data: 音频数据
    :param sample_rate: 采样率
    :param file: 音频文件路径
    :param label_file: 标签文件路径
    :param start_time: 开始时间（秒），如果为 None，则绘制完整图像
    :param end_time: 结束时间（秒），如果为 None，则绘制完整图像
    """
    # 提取婴儿名字
    baby_name = file.split('\\')[1]
    directory = f"output/{baby_name}"
    ensure_directory_exists(directory)

    # 计算MFCC特征随时间的变化
    time, mfccs = calculate_mfcc(data, sample_rate, start_time, end_time)

    # 读取标签文件
    try:
        with open(label_file, 'r') as f:
            segments = [tuple(map(float, line.strip().split())) for line in f]
    except FileNotFoundError:
        print(f"Label file not found: {label_file}")
        return

    # 创建子图
    n_mfcc = mfccs.shape[0]
    fig, axes = plt.subplots(n_mfcc, 1, figsize=(40, 4 * n_mfcc))  # 每个MFCC系数一个子图
    if n_mfcc == 1:
        axes = [axes]  # 如果只有一个MFCC系数，确保axes是列表

    # 遍历每个MFCC系数，绘制子图
    for i in range(n_mfcc):
        ax = axes[i]
        # 分段绘制MFCC特征图
        for seg_start, seg_end, label in segments:
            # 只绘制与给定时间段重叠的部分
            overlap_start = max(seg_start, start_time if start_time is not None else 0)
            overlap_end = min(seg_end, end_time if end_time is not None else time[-1])
            if overlap_start < overlap_end:
                overlap_indices = (time >= overlap_start) & (time <= overlap_end)
                overlap_time = time[overlap_indices]
                if label == 1:  # 标签为1的部分
                    ax.plot(overlap_time, mfccs[i, overlap_indices], color='red', linewidth=1.5, label='Label 1')
                else:  # 其他部分
                    ax.plot(overlap_time, mfccs[i, overlap_indices], color='green', linewidth=1.5, label='Label 0')

        # 设置子图标题和标签
        ax.set_title(f'MFCC {i+1}', fontsize=24)
        ax.set_xlabel('Time [s]', fontsize=20)
        ax.set_ylabel('MFCC Value', fontsize=20)

        # 格式化横轴
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = time[-1]
        xticks = np.arange(start_time, end_time + 1, 20)  # 每 20 秒一个刻度
        xtick_labels = [f"{int(tick // 60):02d}:{int(tick % 60):02d}" for tick in xticks]  # 转换为分钟:秒格式
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)

        # 设置图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # 去重
        ax.legend(by_label.values(), by_label.keys(), fontsize=18)

    # 调整子图间距
    plt.tight_layout()

    # 保存图像
    plt.savefig(f"{directory}/audio_mfcc.png", dpi=300)  # 提高分辨率
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

        # start_time = 100 # 设置为 None 以绘制完整图像
        # end_time = 300  # 设置为 None 以绘制完整图像

        # 绘制MFCC特征图
        plot_mfcc(data, sample_rate, file, label_file, start_time=start_time, end_time=end_time)