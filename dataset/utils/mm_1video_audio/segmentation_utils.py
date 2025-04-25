#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:segmentation_utils.py
# author:xm
# datetime:2024/4/23 10:50
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import cv2
import os
import wave

import pandas as pd

from utils.common_utils import ensure_dir


def split_video(video_path, clip_length=3):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / frame_rate  # Duration of the video in seconds
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an output folder if it doesn't exist
    base_name, ext = os.path.splitext(os.path.basename(video_path))
    output_folder = os.path.join(os.path.dirname(video_path), f'{base_name}_video_clips')
    ensure_dir(output_folder)

    # Initialize variables for splitting
    start_time = 0
    clip_num = 1


    files = []
    while start_time <= duration - clip_length:

        # Set the start and end frames for the current clip
        start_frame = int(start_time * frame_rate)
        end_frame = min(int((start_time + clip_length) * frame_rate), total_frames)

        # Initialize video writer for the current clip
        output_path = os.path.join(output_folder, f'clip_{clip_num}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

        # Read and write frames to the current clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                current_frame += 1
            else:
                break

        # add splited files to array
        files.append(os.path.splitext(output_path)[0])

        # Release the video writer and update start time for the next clip
        out.release()
        start_time += clip_length
        clip_num += 1

    # Release the video capture object
    cap.release()
    return files


def split_audio(audio_path, clip_length=3):
    # Open the audio file
    with wave.open(audio_path, 'rb') as audio_file:
        # Get the parameters of the audio file
        params = audio_file.getparams()
        num_channels, samp_width, frame_rate, num_frames = params[:4]
        frame_length = int(clip_length * frame_rate)

        # Create an output folder if it doesn't exist
        base_name, ext = os.path.splitext(os.path.basename(audio_path))
        output_folder = os.path.join(os.path.dirname(audio_path), f'{base_name}_audio_clips')
        ensure_dir(output_folder)

        files = []
        # Read and split audio into clips
        for i in range(num_frames // frame_length):
            start_frame = i * frame_length
            end_frame = min((i + 1) * frame_length, num_frames)

            audio_file.setpos(start_frame)
            frames = audio_file.readframes(end_frame - start_frame)

            # Write the clip to a new file
            output_path = os.path.join(output_folder, f'clip_{i + 1}.wav')
            with wave.open(output_path, 'wb') as clip_file:
                clip_num_frames = end_frame - start_frame
                clip_params = (num_channels, samp_width, frame_rate, clip_num_frames,
                               params[4], params[5])
                clip_file.setparams(clip_params)
                clip_file.writeframes(frames)

            # add splited files to array
            files.append(os.path.splitext(output_path)[0])

        return files


def read_and_clip_labels(file_path, clip_length=3):
    labels = []
    meta_info = pd.read_csv(file_path, sep='\t', header=None, names=['start_time', 'end_time', 'label'])
    total_duration = meta_info['end_time'].iloc[-1]

    start_info_ptr, end_info_ptr = 0, 0
    current_label = 0 if meta_info['label'].iloc[start_info_ptr] == 0 else 1
    last_cry = 0 if current_label == 1 else -1
    window_start_time = 0
    while window_start_time <= total_duration - clip_length:
        window_end_time = window_start_time + clip_length
        # Check if the current window overlaps with the label interval
        while start_info_ptr < len(meta_info) and meta_info['end_time'].iloc[start_info_ptr] \
                < window_start_time:
            start_info_ptr += 1
            if last_cry < start_info_ptr < len(meta_info):
                current_label = 0

        while end_info_ptr < len(meta_info) and meta_info['end_time'].iloc[end_info_ptr] <= window_end_time:
            end_info_ptr += 1
            if end_info_ptr < len(meta_info) and meta_info['label'].iloc[end_info_ptr] != 0:
                last_cry = end_info_ptr
            if start_info_ptr <= last_cry <= end_info_ptr:
                current_label = 1

        labels.append(current_label)
        window_start_time += clip_length

    return labels


if __name__ == '__main__':
    # Example usage
    video_path = 'test.mp4'
    # audio_path = 'test.wav'
    # label_path = 'test.txt'
    split_video(video_path, clip_length=3)
    # split_audio(audio_path, clip_length=3)
    # read_and_clip_labels(label_path, clip_length=3)