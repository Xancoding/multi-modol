import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm  # 导入 tqdm 库
import pandas as pd


def read_video(file_path):
    """
    读取视频文件并返回视频对象。
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {file_path}.")
        return None
    return cap


def ensure_directory_exists(directory):
    """
    检查文件夹是否存在，如果不存在则创建。
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def visualize_flow(flow, frame, step=16):
    # Create a figure for displaying
    h, w = frame.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Create image and draw
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for (x, y), (fx, fy) in zip(np.stack((x, y), axis=-1), np.stack((fx, fy), axis=-1)):
        cv2.line(vis, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

    return vis


def visualize_frequency_domain(gray):
    f_transform = np.fft.fftshift(np.fft.fft2(gray))
    magnitude_spectrum = np.log(np.abs(f_transform) + 1)
    # Normalize for display
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude_spectrum


def moving_average(frames, kernel_size=5):
    if len(frames) < kernel_size:
        return frames[-1]  # return the latest frame if not enough for averaging
    return np.mean(frames[-kernel_size:], axis=0).astype(np.uint8)  # averaging


def process_video(video_path, kernel_size=5):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frames = []  # Initialize frame buffer for moving average

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        averaged_gray = moving_average(frames, kernel_size)

        # Compute the optical flow using original gray frames
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Visualize the flow using original gray frames
        vis = visualize_flow(flow, gray)
        cv2.imshow('Optical Flow', vis)

        # Visualize the frequency domain using averaged gray frames
        freq_vis = visualize_frequency_domain(averaged_gray)
        cv2.imshow('Frequency Domain', freq_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update previous frame with original gray for accurate optical flow in the next iteration
        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()


def save_motion_intensities(motion_intensities, output_dir, file_name="motion_intensities"):
    """
    将运动强度数组保存为 .npy 和 .csv 文件。
    """
    # 保存为 .npy 文件
    npy_path = os.path.join(output_dir, f"{file_name}.npy")
    np.save(npy_path, motion_intensities)
    print(f"Motion intensities saved to {npy_path}")


def analyze_motion_intensity(video_path, kernel_size=5):
    """
    分析视频中每一帧的运动强度，并返回运动强度数组。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Could not read video {video_path}.")
        return None

    # 将第一帧转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 初始化帧缓冲区和运动强度数组
    frames = []
    motion_intensities = []

    # 使用 tqdm 显示进度条
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while True:
            # 读取下一帧
            ret, frame = cap.read()
            if not ret:
                break

            # 将当前帧转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 将当前灰度帧添加到帧缓冲区
            frames.append(gray)
            # 对帧缓冲区中的帧进行移动平均，得到平滑后的灰度图像
            averaged_gray = moving_average(frames, kernel_size)

            # 使用 Farneback 算法计算光流
            flow = cv2.calcOpticalFlowFarneback(prev_gray, averaged_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 计算光流的幅度
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # 计算当前帧的运动强度（光流幅度的平均值）
            motion_intensity = np.mean(magnitude)
            # 将运动强度添加到数组中
            motion_intensities.append(motion_intensity)

            # 更新前一帧为当前帧，用于下一轮光流计算
            prev_gray = averaged_gray

            # 更新进度条
            pbar.update(1)

    # 释放视频对象
    cap.release()
    # 返回运动强度数组
    return motion_intensities


def analyze_motion_intensity_origin(video_path):
    """
    分析视频中每一帧的运动强度，并返回运动强度数组。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Could not read video {video_path}.")
        return None

    # 将第一帧转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 初始化运动强度数组
    motion_intensities = []

    # 使用 tqdm 显示进度条
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while True:
            # 读取下一帧
            ret, frame = cap.read()
            if not ret:
                break

            # 将当前帧转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 使用 Farneback 算法计算光流
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 计算光流的幅度
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # 计算当前帧的运动强度（光流幅度的平均值）
            motion_intensity = np.mean(magnitude)
            # 将运动强度添加到数组中
            motion_intensities.append(motion_intensity)

            # 更新前一帧为当前帧，用于下一轮光流计算
            prev_gray = gray

            # 更新进度条
            pbar.update(1)

    # 释放视频对象
    cap.release()
    # 返回运动强度数组
    return motion_intensities


def plot_motion_intensities(motion_intensities, output_dir, file_name="video_optical_flow", label_file=None, highlight_crying=True, frame_rate=30, start_time=None, end_time=None):
    """
    绘制运动强度数组的折线图，并格式化横轴为时间轴。
    :param motion_intensities: 运动强度数组
    :param output_dir: 输出目录
    :param file_name: 输出文件名（不含扩展名）
    :param label_file: 标签文件路径
    :param highlight_crying: 是否将哭的部分的线变为红色，默认为 True
    :param frame_rate: 视频帧率（默认 30 FPS）
    :param start_time: 开始时间（秒）
    :param end_time: 结束时间（秒）
    """
    # 确保输出目录存在
    ensure_directory_exists(output_dir)

    # 计算时间轴
    time = np.arange(len(motion_intensities)) / frame_rate  # 将帧索引转换为时间（秒）

    # 如果指定了时间段，则只绘制该时间段的数据
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * frame_rate)
        end_idx = int(end_time * frame_rate)
        time = time[start_idx:end_idx]
        motion_intensities = motion_intensities[start_idx:end_idx]
    else:
        start_time = 0
        end_time = time[-1]

    # 绘制折线图
    plt.figure(figsize=(40, 16))  # 增大图像大小

    if highlight_crying and label_file:
        # 读取标签文件
        with open(label_file, 'r') as f:
            segments = [tuple(map(float, line.strip().split())) for line in f]

        # 分段绘制运动强度
        for seg_start, seg_end, label in segments:
            # 只绘制与给定时间段重叠的部分
            overlap_start = max(seg_start, start_time)
            overlap_end = min(seg_end, end_time)
            if overlap_start < overlap_end:
                overlap_start_idx = int(overlap_start * frame_rate) - int(start_time * frame_rate)
                overlap_end_idx = int(overlap_end * frame_rate) - int(start_time * frame_rate)
                overlap_time = time[overlap_start_idx:overlap_end_idx]
                overlap_motion = motion_intensities[overlap_start_idx:overlap_end_idx]
                if label == 1:  # 哭的部分
                    plt.plot(overlap_time, overlap_motion, color='red', linewidth=1.5)  # 加粗红色线
                else:  # 非哭的部分
                    plt.plot(overlap_time, overlap_motion, color='blue', linewidth=1.5)  # 加粗蓝色线
    else:
        # 如果不标记哭的部分，直接绘制整个运动强度
        plt.plot(time, motion_intensities, color='blue', linewidth=1.5)  # 加粗蓝色线

    plt.title(f'Motion Intensity', fontsize=52)
    plt.xlabel('Time [s]', fontsize=50)
    plt.ylabel('Motion Intensity', fontsize=50)

    # 格式化横轴
    xticks = np.arange(start_time, end_time + 1, 20)  # 每 10 秒一个刻度
    xtick_labels = [f"{int(tick // 60):02d}:{int(tick % 60):02d}" for tick in xticks]  # 转换为分钟:秒格式
    plt.xticks(xticks, xtick_labels, fontsize=50)
    plt.yticks(fontsize=50)

    # 保存图像
    plt.savefig(f"{output_dir}/{file_name}.png", dpi=300)  # 提高分辨率
    plt.close()  # 关闭图像，避免内存泄漏


def plot_motion_intensities_std(motion_intensities, output_dir, file_name="video_optical_flow_std", window_size=30, frame_rate=30, start_time=None, end_time=None, label_file=None, highlight_crying=True):
    """
    绘制运动强度标准差随时间的变化，并根据标签文件对折线图不同部分标色。
    :param motion_intensities: 运动强度数组
    :param output_dir: 输出目录
    :param file_name: 输出文件名（不含扩展名）
    :param window_size: 滑动窗口大小（帧数）
    :param frame_rate: 视频帧率（默认 30 FPS）
    :param start_time: 开始时间（秒）
    :param end_time: 结束时间（秒）
    :param label_file: 标签文件路径
    :param highlight_crying: 是否根据标签标色，默认为 True
    """
    # 确保输出目录存在
    ensure_directory_exists(output_dir)

    # 如果指定了时间段，则只分析该时间段的数据
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * frame_rate)
        end_idx = int(end_time * frame_rate)
        motion_intensities = motion_intensities[start_idx:end_idx]
    else:
        start_time = 0
        end_time = len(motion_intensities) / frame_rate

    # 使用滑动窗口计算标准差
    motion_std = []
    for i in range(len(motion_intensities) - window_size + 1):
        window = motion_intensities[i:i + window_size]
        motion_std.append(np.std(window))

    # 计算时间轴
    time = np.arange(len(motion_std)) / frame_rate + start_time

    # 绘制折线图
    plt.figure(figsize=(40, 16))  # 增大图像大小

    if highlight_crying and label_file:
        # 读取标签文件
        with open(label_file, 'r') as f:
            segments = [tuple(map(float, line.strip().split())) for line in f]

        # 分段绘制运动强度标准差
        for seg_start, seg_end, label in segments:
            # 只绘制与给定时间段重叠的部分
            overlap_start = max(seg_start, start_time)
            overlap_end = min(seg_end, end_time)
            if overlap_start < overlap_end:
                # 计算重叠部分的索引
                overlap_start_idx = int((overlap_start - start_time) * frame_rate)
                overlap_end_idx = int((overlap_end - start_time) * frame_rate)
                overlap_time = time[overlap_start_idx:overlap_end_idx]
                overlap_std = motion_std[overlap_start_idx:overlap_end_idx]
                if label == 1:  # 哭的部分
                    plt.plot(overlap_time, overlap_std, color='red', linewidth=1.5)  # 红色线
                else:  # 非哭的部分
                    plt.plot(overlap_time, overlap_std, color='blue', linewidth=1.5)  # 蓝色线
    else:
        # 如果不标记哭的部分，直接绘制整个运动强度标准差
        plt.plot(time, motion_std, color='green', linewidth=1.5)  # 绿色线

    plt.title(f'Motion Intensity Standard Deviation (Window Size: {window_size} frames)', fontsize=52)
    plt.xlabel('Time [s]', fontsize=50)
    plt.ylabel('Standard Deviation', fontsize=50)

    # 格式化横轴
    xticks = np.arange(start_time, end_time + 1, 20)  # 每 20 秒一个刻度
    xtick_labels = [f"{int(tick // 60):02d}:{int(tick % 60):02d}" for tick in xticks]  # 转换为分钟:秒格式
    plt.xticks(xticks, xtick_labels, fontsize=50)
    plt.yticks(fontsize=50)

    # 保存图像
    plt.savefig(f"{output_dir}/{file_name}.png", dpi=300)  # 提高分辨率
    plt.close()  # 关闭图像，避免内存泄漏漏


if __name__ == "__main__":
    # 查找所有视频文件
    files = glob.glob("infant/*/*cam1.avi")
    for file in files:
        print(f"Processing video: {file}")
        label_file = f"{file.split('_cam1.avi')[0]}_label.txt"

        # 提取婴儿名字
        baby_name = file.split('\\')[1]
        output_dir = f"output/{baby_name}/"
        ensure_directory_exists(output_dir)

        # 分析视频并获取运动强度数组
        # motion_intensities = analyze_motion_intensity_origin(file)
        # motion_intensities = analyze_motion_intensity(file)

        # 保存运动强度数组
        motion_intensities = np.load(f"{output_dir}/motion_intensities_origin.npy")
        # save_motion_intensities(motion_intensities, output_dir, file_name="motion_intensities_origin")

        # 定义时间段（例如：1分钟40秒到2分钟20秒）
        # start_time = 300  # 设置为 None 以绘制完整图像
        # end_time = 500  # 设置为 None 以绘制完整图像
        #
        start_time = 100  # 设置为 None 以绘制完整图像
        end_time = 300  # 设置为 None 以绘制完整图像
        # plot_motion_intensities(motion_intensities, output_dir, file_name="video_optical_flow_origin", label_file=label_file, highlight_crying=False, start_time=start_time, end_time=end_time)
        plot_motion_intensities_std(motion_intensities, output_dir, file_name="video_optical_flow_std_origin", window_size=30, start_time=start_time, end_time=end_time, label_file=label_file, highlight_crying=True)
