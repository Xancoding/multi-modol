# import module your need
import os

import cv2
import librosa
import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def select_roi(frame):
    r = cv2.selectROI("Select ROI", frame)
    cv2.destroyWindow("Select ROI")
    return r  # (x, y, w, h)


def plot_spectrogram(signal, sr, start_sec, title, output_folder):
    plt.figure(figsize=(10, 4))
    nperseg = max(15, len(signal) // 8)  # Adjust window size based on signal length
    f, t, Sxx = spectrogram(signal, sr, nperseg=nperseg)
    # Convert power spectrum to dB scale
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    # Set a consistent color scale for all spectrograms
    vmin = -100  # Minimum dB value for color scale
    vmax = 0  # Maximum dB value for color scale

    plt.pcolormesh(t + start_sec, f, Sxx_dB, shading='gouraud', vmin=vmin, vmax=vmax)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    cbar = plt.colorbar(label='Intensity [dB]')
    cbar.set_label('Intensity [dB]', rotation=270, labelpad=15)

    file_name = f"{output_folder}/spectrogram_{int(start_sec)}_to_{int(start_sec + len(signal) / sr)}.png"
    plt.savefig(file_name)
    plt.close()


def process_video_signal(video_path, output_folder):
    output_folder = f'{output_folder}_video'
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    if not ret:
        return

    roi = select_roi(frame)
    x, y, w, h = roi
    video_signal = []
    start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi_frame = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        video_signal.append(np.mean(gray))

        if len(video_signal) >= 3 * fps:  # Every 3 seconds
            plot_spectrogram(np.array(video_signal), fps, start_time, "Video Signal Spectrogram", output_folder)
            start_time += 3  # Increment start time for next segment
            video_signal = []  # Reset signal for next segment

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def process_audio_signal(audio_path, output_folder):
    output_folder = f'{output_folder}_audio'
    os.makedirs(output_folder, exist_ok=True)

    # Load audio with a target sampling rate of 8000 Hz
    audio_signal, sr = librosa.load(audio_path, sr=8000)
    segment_length = 3 * sr  # 3 seconds of audio data

    for start in range(0, len(audio_signal), segment_length):
        end = start + segment_length
        if end > len(audio_signal):
            end = len(audio_signal)
        segment_signal = audio_signal[start:end]
        plot_spectrogram(segment_signal, sr, start / sr, "Audio Signal Spectrogram", output_folder)


if __name__ == '__main__':
    video_path = '2024-05-26-15-52-06_spx_m.mp4'
    audio_path = '2024-04-27-16-33-04_wxy-f.wav'
    output_folder = './output_spectrograms'

    process_video_signal(video_path, output_folder)
    # process_audio_signal(audio_path, output_folder)
