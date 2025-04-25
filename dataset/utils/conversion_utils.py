#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:conversion_utils.py
# author:xm
# datetime:2024/3/16 15:55
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os
import wave
import numpy as np
import cv2

VIDEO_FPS = 15  # 视频帧率
AUDIO_FPS = 25  # 音频帧率
NUM_PER_AU_FRM = 1920

NUM_CHANNELS = 1  # 声道数
SAMPLE_WIDTH = 2  # 采样位宽（字节）
SAMPLE_RATE = AUDIO_FPS * NUM_PER_AU_FRM  # 采样率
IMG_WIDTH = 800
IMG_HEIGHT = 450
IMG_SIZE = int(IMG_WIDTH * IMG_HEIGHT * 3 / 2)

Y_WIDTH = IMG_WIDTH
Y_HEIGHT = IMG_HEIGHT
Y_SIZE = int(Y_WIDTH * Y_HEIGHT)

U_V_WIDTH = int(IMG_WIDTH / 2)
U_V_HEIGHT = int(IMG_HEIGHT / 2)
U_V_SIZE = int(U_V_WIDTH * U_V_HEIGHT)


def pcm_to_wav(pcm_file, wav_file, num_channels, sample_width, sample_rate):
    pcm_data = np.fromfile(pcm_file, dtype=np.int16)
    with wave.open(wav_file, 'wb') as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data.tobytes())


def from_I420(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_start = y_start + Y_SIZE
        v_start = u_start + U_V_SIZE
        v_end = v_start + U_V_SIZE

        Y[frame_idx, :, :] = yuv_data[y_start: u_start].reshape((Y_HEIGHT, Y_WIDTH))
        U[frame_idx, :, :] = yuv_data[u_start: v_start].reshape((U_V_HEIGHT, U_V_WIDTH))
        V[frame_idx, :, :] = yuv_data[v_start: v_end].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_YV12(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        v_start = y_start + Y_SIZE
        u_start = v_start + U_V_SIZE
        u_end = u_start + U_V_SIZE

        Y[frame_idx, :, :] = yuv_data[y_start: v_start].reshape((Y_HEIGHT, Y_WIDTH))
        V[frame_idx, :, :] = yuv_data[v_start: u_start].reshape((U_V_HEIGHT, U_V_WIDTH))
        U[frame_idx, :, :] = yuv_data[u_start: u_end].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_NV12(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_v_start = y_start + Y_SIZE
        u_v_end = u_v_start + (U_V_SIZE * 2)

        Y[frame_idx, :, :] = yuv_data[y_start: u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
        U_V = yuv_data[u_v_start: u_v_end].reshape((U_V_SIZE, 2))
        U[frame_idx, :, :] = U_V[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
        V[frame_idx, :, :] = U_V[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_NV21(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_v_start = y_start + Y_SIZE
        u_v_end = u_v_start + (U_V_SIZE * 2)

        Y[frame_idx, :, :] = yuv_data[y_start: u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
        U_V = yuv_data[u_v_start: u_v_end].reshape((U_V_SIZE, 2))
        V[frame_idx, :, :] = U_V[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
        U[frame_idx, :, :] = U_V[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def np_yuv2rgb(Y, U, V):
    bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    V = np.repeat(V, 2, 0)
    V = np.repeat(V, 2, 1)
    U = np.repeat(U, 2, 0)
    U = np.repeat(U, 2, 1)

    c = (Y - np.array([16])) * 298
    d = U - np.array([128])
    e = V - np.array([128])

    r = (c + 409 * e + 128) // 256
    g = (c - 100 * d - 208 * e + 128) // 256
    b = (c + 516 * d + 128) // 256

    # 修剪值以确保它们在0到255的范围内
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    bgr_data[:, :, 2] = r
    bgr_data[:, :, 1] = g
    bgr_data[:, :, 0] = b

    return bgr_data


def yuv2rgb(Y, U, V):
    bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    for h_idx in range(Y_HEIGHT):
        for w_idx in range(Y_WIDTH):
            y = Y[h_idx, w_idx]
            u = U[int(h_idx // 2), int(w_idx // 2)]
            v = V[int(h_idx // 2), int(w_idx // 2)]

            c = (y - 16) * 298
            d = u - 128
            e = v - 128

            r = (c + 409 * e + 128) // 256
            g = (c - 100 * d - 208 * e + 128) // 256
            b = (c + 516 * d + 128) // 256

            bgr_data[h_idx, w_idx, 2] = 0 if r < 0 else (255 if r > 255 else r)
            bgr_data[h_idx, w_idx, 1] = 0 if g < 0 else (255 if g > 255 else g)
            bgr_data[h_idx, w_idx, 0] = 0 if b < 0 else (255 if b > 255 else b)

    return bgr_data


def calculate_num_frames(filename):
    # calc file size
    file_size = os.path.getsize(filename)
    # count frames
    num_frames = file_size // IMG_SIZE
    return num_frames


# convert nv12 to rgb, by numpy
def convert_nv12_to_rgb_video(filename, width, height, output_filename, fps=30):
    # count the number of video frames
    num_frames = calculate_num_frames(filename)
    print(f'frame_num of video: {num_frames}')

    # set params of output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    with open(filename, "rb") as yuv_f:
        yuv_bytes = yuv_f.read()
        yuv_data = np.frombuffer(yuv_bytes, np.uint8)

        Y, U, V = from_NV12(yuv_data, num_frames)

        # rgb_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        for frame_idx in range(num_frames):
            # bgr_data = yuv2rgb(Y[frame_idx, :, :], U[frame_idx, :, :], V[frame_idx, :, :])            # for
            bgr_data = np_yuv2rgb(Y[frame_idx, :, :], U[frame_idx, :, :], V[frame_idx, :, :])  # numpy
            if bgr_data is not None:
                # write rgb to mp4
                writer.write(bgr_data)
                frame_idx += 1

    # release resource
    writer.release()


# convert nv12 to rgb by opencv
def convert_nv12_to_rgb(input_file, output_file, fps):
    # read bin file
    with open(input_file, 'rb') as f:
        video_bytes = f.read()

    # count the number of video frames
    num_frames = calculate_num_frames(input_file)

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (IMG_WIDTH, IMG_HEIGHT))

    # iter each frame
    for i in range(num_frames):
        # extract data of current frame
        start = i * IMG_SIZE
        end = start + IMG_SIZE
        frame_bytes = video_bytes[start:end]

        # convert bin data to yuv
        yuv = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(IMG_HEIGHT * 3 // 2, IMG_WIDTH)

        # conversion
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

        # write to out file
        out.write(bgr)

    # release resource
    out.release()


if __name__ == '__main__':
    file_name = '2024-06-02-15-58-44_llr_f'
    pcm_file = f'{file_name}.pcm'  # input pcm
    wav_file = f'{file_name}.wav'  # out wav

    yuv_file = f'{file_name}.bin'  # input yuv
    mp4_file = f'{file_name}.mp4'  # out mp4

    # convert_nv12_to_rgb_video(yuv_file, IMG_WIDTH, IMG_HEIGHT, mp4_file, VIDEO_FPS)
    convert_nv12_to_rgb(yuv_file, mp4_file, VIDEO_FPS)
    pcm_to_wav(pcm_file, wav_file, NUM_CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE)

    print("Conversion complete.")
