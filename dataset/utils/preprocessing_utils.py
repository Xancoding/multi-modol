#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:preprocessing_utils.py
# author:xm
# datetime:2024/4/23 14:37
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os
import shutil
from collections import defaultdict, Counter

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedGroupKFold

import config
from dataset.utils.segmentation_utils import split_audio, split_video, read_and_clip_labels
from utils.common_utils import ensure_dir


def get_mel_spectrogram_transform(args):
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=args.n_mels,
        mel_scale="slaney",
    )
    return mel_spectrogram_transform


def spectrogram_padding(sepctrogram, max_len):
    assert sepctrogram.shape[-1] < max_len, "can't be greater than max length"
    padding = max_len - sepctrogram.shape[-1]
    sepctrogram = F.pad(sepctrogram, (0, padding), "constant", value=0)
    return sepctrogram


def extract_mel_spectrogram(waveform, mel_spectrogram_transform, args):
    """
    :param audio_file:
    :param mel_spectrogram_transform:
    :return: log_mel_spec: torch.Size([1, n_mels, max_len])
    """
    # extract mel_spectrogram
    mel_spec = mel_spectrogram_transform(waveform)
    # Convert to logarithmic scale
    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    # normalize
    norm_mel = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    # padding
    # norm_mel = spectrogram_padding(norm_mel, args.max_len)

    return norm_mel


def downsample(data, labels, groups):
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    pos_count = len(pos_idx)
    neg_count = len(neg_idx)

    # encourage the ratio of positive and negative samples close to 1:1
    if neg_count > pos_count:
        desired_neg_count = pos_count
    else:
        # no need to downsample
        return data, labels, groups

    # Randomly select negative samples
    sampled_neg_idx = np.random.choice(neg_idx, desired_neg_count, replace=False)

    # Merge positive samples and downsampled negative samples
    sampled_idx = np.concatenate([pos_idx, sampled_neg_idx])

    return data[sampled_idx], labels[sampled_idx], groups[sampled_idx]

def split_data(args):
    data_dir = args.data_dir
    save_dir = args.cross_val_dir
    ensure_dir(save_dir)

    all_files, all_labels, groups, group = [], [], [], 0

    # Traverse hospitals
    for hospital_name in os.listdir(data_dir):
        hospital_dir = os.path.join(data_dir, hospital_name)
        # Traverse cry and non-cry folders
        for folder_name in ['cry', 'non-cry']:
            folder_dir = os.path.join(hospital_dir, folder_name)
            # Traverse Patients
            for patient_name in os.listdir(folder_dir):
                patient_dir = os.path.join(folder_dir, patient_name)
                # Traverse files in patient folder
                video_files_cam0, video_files_cam1, audio_files, labels = [], [], [], []

                tmp_results = defaultdict(lambda: {'audio_files': [], 'video_files_cam0': [], 'video_files_cam1': [], 'labels': []})
                for file_name in os.listdir(patient_dir):
                    file_path = os.path.join(patient_dir, file_name)
                    base_name, ext = os.path.splitext(os.path.basename(file_name))
                    base_name = '_'.join(base_name.split('_')[:2])
                    base_name = base_name.replace('_audio_clips', '').replace('_video_clips', '')
                    # split data
                    audio_clips_dir = os.path.join(patient_dir, f'{base_name}_aduio_audio_clips')
                    video_clips_dir_cam0 = os.path.join(patient_dir, f'{base_name}_cam0_video_clips')
                    video_clips_dir_cam1 = os.path.join(patient_dir, f'{base_name}_cam1_video_clips')

                    if os.path.isdir(file_path):
                        if 'audio_clips' in file_name:
                            tmp_results[base_name]['audio_files'] = [
                                os.path.join(audio_clips_dir, os.path.splitext(fs)[0])
                                for fs in os.listdir(file_path)]
                        elif 'video_clips' in file_name and 'cam0' in file_name:
                            tmp_results[base_name]['video_files_cam0'] = [
                                os.path.join(video_clips_dir_cam0, os.path.splitext(fs)[0])
                                for fs in os.listdir(file_path)]
                        elif 'video_clips' in file_name and 'cam1' in file_name:
                            tmp_results[base_name]['video_files_cam1'] = [
                                os.path.join(video_clips_dir_cam1, os.path.splitext(fs)[0])
                                for fs in os.listdir(file_path)]
                    else:
                        if ext == '.avi' and not os.path.exists(video_clips_dir_cam0) and file_name.endswith('_cam0.avi'):
                            tmp_results[base_name]['video_files_cam0'] = split_video(file_path, clip_length=args.clip_length, resize=args.resize)
                        elif ext == '.avi' and not os.path.exists(video_clips_dir_cam1) and file_name.endswith('_cam1.avi'):
                            tmp_results[base_name]['video_files_cam1'] = split_video(file_path, clip_length=args.clip_length, resize=args.resize)
                        elif ext == '.wav' and not os.path.exists(audio_clips_dir):
                            tmp_results[base_name]['audio_files'] = split_audio(file_path, clip_length=args.clip_length)
                        elif ext == '.txt' and file_name.endswith('_label.txt'): 
                            tmp_results[base_name]['labels'] = \
                                read_and_clip_labels(file_path, clip_length=args.clip_length)

                for val in tmp_results.values():
                    # ensure orders
                    sorted_audios = sorted(val['audio_files'], key=lambda x: int(os.path.basename(x).split('_')[1]))
                    sorted_videos_cam0 = sorted(val['video_files_cam0'], key=lambda x: int(os.path.basename(x).split('_')[1]))
                    sorted_videos_cam1 = sorted(val['video_files_cam1'], key=lambda x: int(os.path.basename(x).split('_')[1]))

                    audio_files += sorted_audios
                    video_files_cam0 += sorted_videos_cam0
                    video_files_cam1 += sorted_videos_cam1
                    if folder_name == 'cry':
                        labels += val['labels']
                    else:
                        labels += [0] * len(val['audio_files'])

                assert len(video_files_cam0) == len(audio_files), f'{hospital_name}: {patient_name},' \
                                                             f' ensure cam0 video and audio are same length'
                assert len(video_files_cam1) == len(audio_files), f'{hospital_name}: {patient_name},' \
                                                             f' ensure cam1 video and audio are same length'
                assert len(video_files_cam0) == len(labels), f'{hospital_name}: {patient_name}, ' \
                                                        f'ensure cam0 video and labels are same length'
                assert len(video_files_cam1) == len(labels), f'{hospital_name}: {patient_name}, ' \
                                                        f'ensure cam1 video and labels are same length'

                all_files += list(zip(audio_files, video_files_cam0, video_files_cam1))
                all_labels += labels
                groups += [group] * len(labels)
                group += 1

    # split dataset
    all_files, all_labels, groups = np.array(all_files), np.array(all_labels), np.array(groups)
    # Downsample negative samples on the training set
    all_files, all_labels, groups = downsample(all_files, all_labels, groups)

    fold = 0
    sgkf = StratifiedGroupKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    for train_val_idx, test_idx in sgkf.split(all_files, all_labels, groups=groups):
        save_train_path = os.path.join(save_dir, f'Fold_{fold}_TrainSet.txt')
        save_val_path = os.path.join(save_dir, f'Fold_{fold}_ValSet.txt')
        save_test_path = os.path.join(save_dir, f'Fold_{fold}_TestSet.txt')

        train_val_files, train_val_labels = all_files[train_val_idx], all_labels[train_val_idx]
        test_files, test_labels = all_files[test_idx], all_labels[test_idx]

        train_idx, val_idx = next(
            StratifiedGroupKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed).split(
                train_val_files, train_val_labels, groups=groups[train_val_idx]
            ))

        train_files, train_labels = train_val_files[train_idx], train_val_labels[train_idx]
        val_files, val_labels = train_val_files[val_idx], train_val_labels[val_idx]

        with open(save_train_path, 'w') as f:
            for sample_idx in range(len(train_idx)):
                f.write(f'{train_files[sample_idx][0]}\t{train_files[sample_idx][1]}\t{train_files[sample_idx][2]}\t{train_labels[sample_idx]}\n')
        with open(save_val_path, 'w') as f:
            for sample_idx in range(len(val_idx)):
                f.write(f'{val_files[sample_idx][0]}\t{val_files[sample_idx][1]}\t{val_files[sample_idx][2]}\t{val_labels[sample_idx]}\n')
        with open(save_test_path, 'w') as f:
            for sample_idx in range(len(test_idx)):
                f.write(f'{test_files[sample_idx][0]}\t{test_files[sample_idx][1]}\t{test_files[sample_idx][2]}\t{test_labels[sample_idx]}\n')

        fold += 1

# def preprocess_video(video_path, resize=(320, 180), fps=15):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)
#     assert int(cap.get(cv2.CAP_PROP_FPS)) % fps == 0, "ensure video fps is a multiple of target fps"

#     sample_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
#     frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     short_crop_size = int(min(frame_width, frame_height) * 0.88)
#     long_crop_size = int(short_crop_size * 16 / 9)
#     frames = []

#     # Read frames from the video
#     current_frame = 0
#     while (cap.isOpened()):
#         ret, frame = cap.read()
#         if ret:
#             if current_frame % sample_interval == 0:
#                 # Crop the frame to the specified size
#                 frame = crop_and_resize(frame, (short_crop_size, long_crop_size), resize)
#                 # Convert the frame to grayscale
#                 gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 # Convert the frame to torch tensor
#                 frame_tensor = frame_to_tensor(gray_frame)
#                 frames.append(frame_tensor)
#             current_frame += 1
#         else:
#             break

#     # Close the video file
#     cap.release()

#     # Stack frames into a single tensor along the time dimension
#     video_tensor = torch.stack(frames)

#     return video_tensor.transpose(0, 1)


def preprocess_video(video_path, resize=(480, 300), fixed_frames=150):
    """强制固定输出帧数的视频预处理
    
    Args:
        video_path: 视频路径
        resize: 目标分辨率 (width, height)
        fixed_frames: 强制输出的固定帧数
    Returns:
        torch.Tensor: 形状为 [1, fixed_frames, H, W] 的张量
    """
    cap = cv2.VideoCapture(video_path)

    # 获取视频属性
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 预分配固定大小的张量 [1, fixed_frames, H, W]
    video_tensor = torch.zeros((1, fixed_frames, resize[1], resize[0]))
    
    # 计算采样策略
    if total_frames >= fixed_frames:
        # 长视频：均匀采样
        step = total_frames / fixed_frames
        sample_positions = [int(i * step) for i in range(fixed_frames)]
    else:
        # 短视频：循环填充
        sample_positions = list(range(total_frames)) * (fixed_frames // total_frames + 1)
        sample_positions = sample_positions[:fixed_frames]
    
    # 读取并处理帧
    valid_frames = 0
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            # 严格尺寸转换
            resized = cv2.resize(frame, resize)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite("gray.png", gray)
            video_tensor[0, valid_frames] = torch.from_numpy(gray).float() / 255.0
            valid_frames += 1
    
    cap.release()
    return video_tensor



def crop_and_resize(frame, crop_size, resize):
    h, w, _ = frame.shape
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    cropped_frame = frame[top:bottom, left:right]
    resized_frame = cv2.resize(cropped_frame, resize, interpolation=cv2.INTER_LINEAR)
    return resized_frame


def frame_to_tensor(frame):
    # Check if the frame is grayscale or color
    if len(frame.shape) == 2:  # Grayscale image
        # Add a channel dimension (at the first position) and convert to CHW format
        frame = frame[None, :, :]  # Same as frame[np.new_axis, :, :]
    else:  # Color image
        # Convert HWC to CHW
        frame = frame.transpose((2, 0, 1))
    frame = torch.tensor(frame, dtype=torch.float32)
    frame = (frame - 128.0) / 128.0
    return frame


def read_data(file_path):
    """
    Read data file containing audio path, video_cam0 path, video_cam1 path, and label
    Format: audio_path\tvideo_cam0_path\tvideo_cam1_path\tlabel
    """
    files, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split line into components
            parts = line.strip().split('\t')
            
            # Handle both old (3 columns) and new (4 columns) formats
            if len(parts) == 3:
                # Old format: audio, video, label
                audio_file, video_file, label = parts
                files.append((audio_file, video_file))
            elif len(parts) == 4:
                # New format: audio, video_cam0, video_cam1, label
                audio_file, video_cam0, video_cam1, label = parts
                files.append((audio_file, video_cam0, video_cam1))
            else:
                raise ValueError(f"Invalid line format in {file_path}: {line.strip()}")
                
            labels.append(int(label))

    return files, labels


def read_feature(file_path):
    """
    Load preprocessed features from a joblib file, supporting both single and dual video streams.
    
    Args:
        file_path (str): Path to the joblib file containing features
        
    Returns:
        tuple: (audio_features, video_features) for single video stream
        or
        tuple: (audio_features, video_features_cam0, video_features_cam1) for dual video streams
        
    Raises:
        KeyError: If required features are not found in the data
    """
    data = joblib.load(file_path)[0]
    
    # Get audio features (required)
    audio_features = data['audio_features']
    
    # Handle both single and dual video stream cases
    if 'video_features_cam0' in data and 'video_features_cam1' in data:
        # Dual video stream case
        return audio_features, data['video_features_cam0'], data['video_features_cam1']
    elif 'video_features' in data:
        # Single video stream case (backward compatible)
        return audio_features, data['video_features']
    else:
        raise KeyError(f"Video features not found in {file_path}. Expected either 'video_features' "
                      f"(single stream) or both 'video_features_cam0' and 'video_features_cam1' "
                      f"(dual streams)")


def preprocess_audio(audio_file, transform, args):
    waveform, sample_rate = torchaudio.load(audio_file)
    # uniform sample rate (default=8 KHz)
    if sample_rate != args.sample_rate:
        waveform = T.Resample(sample_rate, args.sample_rate)(waveform)
        sample_rate = args.sample_rate
    # Mono data only
    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # spectrogram features
    spectrogram = extract_mel_spectrogram(waveform, transform, args)
    return spectrogram


def store_features(args):
    cross_validation_dir = args.cross_val_dir
    feature_dir = args.feature_dir
    ensure_dir(feature_dir)
    mel_spectrogram_transform = get_mel_spectrogram_transform(args)

    # store features (only one fold is enough)
    for file_name in os.listdir(cross_validation_dir):
        if 'Fold_0' not in file_name:
            continue
        f = os.path.join(cross_validation_dir, file_name)
        data_files, labels = read_data(f)

        for idx in range(len(data_files)):
            data = []
            audio_file, video_file, label = data_files[idx][0] + '.wav', data_files[idx][1] + '.mp4', labels[idx]
            # audio processing
            spectrograms = preprocess_audio(audio_file, mel_spectrogram_transform, args)
            # video processing
            video_tensor = preprocess_video(video_file)
            data.append({'audio_features': spectrograms, 'video_features': video_tensor})

            cur_dir = os.path.basename(os.path.dirname(audio_file))
            cur_dir = cur_dir.replace('_audio_clips', '').replace('_video_clips', '')
            clip_id = int(os.path.basename(data_files[idx][0]).split('_')[1])
            save_path = os.path.join(os.path.join(feature_dir, cur_dir), f'clip_{clip_id}.joblib')
            ensure_dir(save_path)
            joblib.dump(data, save_path, compress='zlib')
            print(f'successfully store features at {save_path}')


def clean_data(folder_path, pattern='clips'):
    """
        删除指定文件夹下所有名称中包含"clips"的子文件夹。

        Args:
            folder_path (str): 需要清理的文件夹路径。
    """
    for root, dirs, files in os.walk(folder_path):
        # Traverse all sub_dirs
        for dir in dirs:
            # pattern regularization
            if pattern in dir.lower():
                # construct the path of subdir
                sub_dir_path = os.path.join(root, dir)
                # delete sub dir
                shutil.rmtree(sub_dir_path, ignore_errors=True)
                print(f"dir: {sub_dir_path} is deleted")


if __name__ == '__main__':
    args = config.get_config()
    args.data_dir = '../../' + args.data_dir
    args.cross_val_dir = '../../' + args.cross_val_dir
    args.feature_dir = '../../' + args.feature_dir
    # split_data(args)
    # store_features(args)
    # preprocess_video('test.mp4')
    _, labels = read_data('../../data/cross_validation/Fold_0_TrainSet.txt')
    labels += read_data('../../data/cross_validation/Fold_0_ValSet.txt')[1]
    labels += read_data('../../data/cross_validation/Fold_0_TestSet.txt')[1]
    counter = Counter(labels)
    print(counter.get(0), counter.get(1))
