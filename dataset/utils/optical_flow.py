#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:optical_flow.py
# author:xm
# datetime:2024/9/6 17:34
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import cv2
import numpy as np


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


if __name__ == '__main__':
    process_video('2024-04-27-16-33-04_wxy-f.mp4')
