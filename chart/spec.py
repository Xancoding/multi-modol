import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import matplotlib.mlab as mlab
from datetime import timedelta

def read_wav(file_path):
    """读取WAV文件并转为单声道"""
    rate, data = wavfile.read(file_path)
    return rate, data.mean(axis=1) if data.ndim == 2 else data

def sec_to_hms(seconds):
    """秒数转HH:MM:SS.SSS格式"""
    td = timedelta(seconds=seconds)
    hms = str(td).split('.')[0]
    ms = f".{int(round(td.microseconds / 1000)):03d}" if td.microseconds else ".000"
    return hms + ms if '.' not in hms else hms

def hms_to_sec(time_str):
    """HH:MM:SS.SSS格式转秒数"""
    h, m, s = time_str.split(':')
    s_parts = s.split('.')
    seconds = float(h)*3600 + float(m)*60 + float(s_parts[0])
    if len(s_parts) > 1:
        seconds += float(s_parts[1])/1000
    return seconds

def plot_spectrogram(data, rate, start=None, end=None):
    """绘制语谱图"""
    # 计算总时长
    total_duration = len(data) / rate
    
    # 处理时间范围
    start_sec = hms_to_sec(start) if start else 0
    end_sec = hms_to_sec(end) if end else total_duration
    
    # 验证时间范围
    if start_sec < 0:
        start_sec = 0
    if end_sec > total_duration:
        end_sec = total_duration
    if start_sec >= end_sec:
        raise ValueError("开始时间必须早于结束时间")

    # 提取数据段
    start_idx = int(start_sec * rate)
    end_idx = int(end_sec * rate)
    segment = data[start_idx:end_idx]
    
    # 生成语谱图
    spec, freqs, t = mlab.specgram(
        segment,
        Fs=rate,
        NFFT=1024,
        noverlap=512,
        mode='magnitude'
    )
    
    # 创建时间轴
    t = np.linspace(start_sec, end_sec, len(t))
    
    # 绘图
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(t, freqs, 10*np.log10(spec), cmap='viridis', shading='auto')
    plt.colorbar(label='Intensity [dB]')
    plt.title(f'Audio Spectrogram ({sec_to_hms(start_sec)} - {sec_to_hms(end_sec)})')
    plt.xlabel('Time')
    plt.ylabel('Frequency [Hz]')
    
    # 设置刻度
    plt.xticks(
        np.linspace(start_sec, end_sec, 5),
        [sec_to_hms(t) for t in np.linspace(start_sec, end_sec, 5)],
        rotation=45
    )
    plt.yticks(np.linspace(0, freqs[-1], 5))
    
    plt.tight_layout()
    plt.savefig('spec.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"语谱图已保存至: spec.png")
    print(f"时间段: {sec_to_hms(start_sec)} - {sec_to_hms(end_sec)}")
    print(f"总时长: {sec_to_hms(total_duration)}")
    print(f"分析时长: {sec_to_hms(end_sec - start_sec)}")

if __name__ == "__main__":
    # prefix = "/data/Leo/mm/data/Newborn200/"
    # infant = "46cm2.4kg"    
    prefix = "/data/Leo/mm/data/NanfangHospital/"
    infant = "lxm-baby-m_2025-07-29-14-55-30"
    audio_file = f"{prefix}data/{infant}.wav"
    rate, data = read_wav(audio_file)
    plot_spectrogram(data, rate, '00:00:10.000', '00:00:12.500')