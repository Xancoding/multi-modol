import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import os
import matplotlib as mpl
import numpy as np
import matplotlib.mlab as mlab
from datetime import timedelta

# 公共配置（完全按照参考代码设置）
COMMON_CONFIG = {
    'linewidth': 1,
    'linestyle': '-',
    'grid': True,
    'grid_style': {'linestyle': '--', 'alpha': 0.5},
    'title_fontsize': 10,
    'label_fontsize': 9,
    'tick_fontsize': 8,
    'legend_fontsize': 8,
    'figsize': (12, 6),
    'dpi': 300,
    'time_format': '%H:%M:%S'
}

# 语谱图特定配置
SPECTROGRAM_CONFIG = {
    'cmap': 'viridis',
    'nfft': 1024,
    'noverlap': 512,
    'colorbar_label': 'Intensity [dB]',
    'title': 'Audio Spectrogram',
    'xlabel': 'Time',
    'ylabel': 'Frequency [Hz]'
}

def sec_to_hms(seconds):
    """将秒数转换为HH:MM:SS格式（完全匹配参考代码）"""
    return str(timedelta(seconds=seconds)).split('.')[0]

def hms_to_sec(time_str):
    """将HH:MM:SS格式转换为秒数（完全匹配参考代码）"""
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

def read_wav_file(file_path):
    """读取WAV文件"""
    sample_rate, data = wavfile.read(file_path)
    if data.ndim == 2:
        data = data.mean(axis=1)  # 对左右声道取平均
    return sample_rate, data

def plot_spectrogram(data, sample_rate, file_path, start_time=None, end_time=None):
    """
    绘制音频语谱图（完全按照参考代码风格）
    :param data: 音频数据
    :param sample_rate: 采样率
    :param file_path: 音频文件路径
    :param start_time: 开始时间（HH:MM:SS格式），None表示从头开始
    :param end_time: 结束时间（HH:MM:SS格式），None表示到结尾结束
    """
    # 计算总时长
    total_duration = len(data) / sample_rate
    
    # 处理时间范围（完全匹配参考代码逻辑）
    start_sec = hms_to_sec(start_time) if start_time else 0
    end_sec = hms_to_sec(end_time) if end_time else total_duration
    
    # 验证时间范围
    if start_sec < 0:
        print(f"警告: 开始时间 {start_time} 早于音频开始时间，将使用00:00:00")
        start_sec = 0
    if end_sec > total_duration:
        print(f"警告: 结束时间 {end_time} 晚于音频结束时间({sec_to_hms(total_duration)})，将使用音频结束时间")
        end_sec = total_duration
    if start_sec >= end_sec:
        raise ValueError(f"错误: 开始时间 {start_time} 不早于结束时间 {end_time}")

    # 提取数据段
    start_idx = int(start_sec * sample_rate)
    end_idx = int(end_sec * sample_rate)
    segment_data = data[start_idx:end_idx]

    # 创建画布（尺寸和参考代码完全一致）
    fig, ax = plt.subplots(figsize=COMMON_CONFIG['figsize'])
    
    # 生成语谱图（参数与参考代码风格一致）
    spec, freqs, t = mlab.specgram(
        segment_data,
        Fs=sample_rate,
        NFFT=SPECTROGRAM_CONFIG['nfft'],
        noverlap=SPECTROGRAM_CONFIG['noverlap']
    )
    
    # 调整时间轴
    t += start_sec
    t = t[:-1]  # 去掉最后一列
    spec = spec[:, :-1]

    # 绘制语谱图（颜色映射等与参考代码风格一致）
    im = ax.pcolormesh(t, freqs, 10 * np.log10(spec), 
                      cmap=SPECTROGRAM_CONFIG['cmap'], 
                      shading='auto')
    
    # 添加颜色条（样式与参考代码一致）
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(SPECTROGRAM_CONFIG['colorbar_label'], 
                  fontsize=COMMON_CONFIG['label_fontsize'])

    # 设置标题和标签（字体大小与参考代码完全一致）
    ax.set_title(SPECTROGRAM_CONFIG['title'], 
                fontsize=COMMON_CONFIG['title_fontsize'], 
                pad=10)
    ax.set_xlabel(SPECTROGRAM_CONFIG['xlabel'], 
                 fontsize=COMMON_CONFIG['label_fontsize'])
    ax.set_ylabel(SPECTROGRAM_CONFIG['ylabel'], 
                 fontsize=COMMON_CONFIG['label_fontsize'])

    # 设置刻度（完全按照参考代码的样式）
    time_labels = [sec_to_hms(sec) for sec in t]
    tick_positions = np.linspace(0, len(t)-1, min(5, len(t)), dtype=int)
    
    ax.set_xticks(t[tick_positions])
    ax.set_xticklabels([time_labels[i] for i in tick_positions], 
                      fontsize=COMMON_CONFIG['tick_fontsize'])
    
    # 频率轴刻度（与参考代码Y轴刻度风格一致）
    yticks = np.linspace(0, freqs[-1], 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(freq)}" for freq in yticks], 
                       fontsize=COMMON_CONFIG['tick_fontsize'])

    # 设置坐标轴范围
    ax.set_xlim(start_sec, end_sec)
    ax.set_ylim(0, freqs[-1])

    # 添加网格（样式与参考代码完全一致）
    if COMMON_CONFIG['grid']:
        ax.grid(**COMMON_CONFIG['grid_style'])

    # 设置边框样式（与参考代码完全一致）
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#DDDDDD')
        spine.set_linewidth(0.8)

    # 调整布局（与参考代码一致）
    plt.tight_layout()

    # 保存图像（DPI等参数与参考代码一致）
    output_path= 'img/audio_spectrogram.jpg'
    
    plt.savefig(output_path, dpi=COMMON_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"成功生成语谱图: {output_path}")
    print(f"时间段: {sec_to_hms(start_sec)} - {sec_to_hms(end_sec)}")
    print(f"音频总时长: {sec_to_hms(total_duration)}")

if __name__ == "__main__":
    # # 查找所有WAV文件（与参考代码风格一致）
    # prefix = "/data/Leo/mm/data/raw_data/NanfangHospital/cry/drz-m/"
    # files = glob.glob(os.path.join(prefix, "*.wav"))

    prefix = "/data/Leo/mm/data/raw_data/"
    temp_files = [
        # "NanfangHospital/cry/drz-m",
        # "ShenzhenUniversityGeneralHospital/non-cry/zlj-baby-f",
        "NanfangHospital/cry/llj-baby-f",
        # "NanfangHospital/non-cry/lyz-m",
    ]
    dirs = [prefix + file + "/*.wav" for file in temp_files]
    files = [glob.glob(d) for d in dirs]
    files = [item for sublist in files for item in sublist]

    # prefix = '/data/Leo/mm/data/Newborn200'
    # temp_files = [
    #     # '/Cry/features/48cm2.66kg',
    #     # '/Cry/features/buduiqi_50cm3.4kg2',
    #     '/NoCry/50cm3.18kg2'
    # ]
    # files = [prefix + file + '.wav' for file in temp_files]

    files = ['a.wav']
    for file in files:
        print(f"\n处理音频文件: {file}")
        
        # 读取WAV文件
        sample_rate, data = read_wav_file(file)
        # print("Max amplitude:", np.max(np.abs(data)))  # 应大于0.001


        # plot_spectrogram(data, sample_rate, file, start_time="00:00:14", end_time="00:00:50")
        # plot_spectrogram(data, sample_rate, file, start_time="00:04:30", end_time="00:05:00")
        plot_spectrogram(data, sample_rate, file)