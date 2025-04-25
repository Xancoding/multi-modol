import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import os
import matplotlib as mpl
import numpy as np  # 确保导入 numpy
import matplotlib.mlab as mlab

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

def plot_spectrogram(data, sample_rate, file, start_time=None, end_time=None):
    """
    绘制音频语谱图，并格式化横轴为时间轴。
    :param data: 音频数据
    :param sample_rate: 采样率
    :param file: 音频文件路径
    :param start_time: 开始时间（秒），如果为 None，则绘制完整图像
    :param end_time: 结束时间（秒），如果为 None，则绘制完整图像
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
    segment_data = data[start_idx:end_idx]

    # 绘制语谱图
    plt.figure(figsize=(40, 16))  # 增大图像大小

    # 生成语谱图
    spec, freqs, t = mlab.specgram(segment_data, Fs=sample_rate, NFFT=1024, noverlap=512)

    # 调整 t 轴，使其从 start_time 开始
    t += start_time

    # 去掉最右边的那一条（最后一列数据）
    t = t[:-1]
    spec = spec[:, :-1]

    # 绘制语谱图
    plt.pcolormesh(t, freqs, 10 * np.log10(spec), cmap='viridis')
    plt.colorbar(label='Intensity [dB]', pad=0.02)

    # 设置标题和标签
    plt.title(f'Spectrogram', fontsize=52)
    plt.xlabel('Time [s]', fontsize=50)
    plt.ylabel('Frequency [Hz]', fontsize=50)

    # 格式化横轴
    xticks = np.arange(start_time, end_time + 1, 20)  # 每 20 秒一个刻度
    xtick_labels = [f"{int(tick // 60):02d}:{int(tick % 60):02d}" for tick in xticks]  # 转换为分钟:秒格式
    plt.xticks(xticks, xtick_labels, fontsize=50)
    plt.yticks(fontsize=50)

    # 设置 x 轴范围，避免空白区域
    plt.xlim(start_time, end_time)

    # 保存图像
    plt.savefig(f"{directory}/audio_spectrogram.png", dpi=300)  # 提高分辨率
    plt.close()  # 关闭图像，避免内存泄漏


if __name__ == "__main__":
    # 查找所有 WAV 文件
    files = glob.glob("infant/*/*.wav")
    for file in files:
        print(f"Processing audio: {file}")
        # 读取 WAV 文件
        sample_rate, data = read_wav_file(file)

        # 定义时间段（例如：1分钟40秒到2分钟20秒）
        start_time = 300  # 设置为 None 以绘制完整图像
        end_time = 500  # 设置为 None 以绘制完整图像
        #
        # start_time = 100  # 设置为 None 以绘制完整图像
        # end_time = 300  # 设置为 None 以绘制完整图像

        # 绘制语谱图
        plot_spectrogram(data, sample_rate, file, start_time, end_time)