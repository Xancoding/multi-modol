#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:basic_dataset.py
# author:xm
# datetime:2024/4/23 11:10
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from typing import Tuple

from torch.utils.data import Dataset, DataLoader
import os

import config
from dataset.utils.preprocessing_utils import read_data, get_mel_spectrogram_transform, preprocess_audio, \
    preprocess_video, read_feature


class VideoAudioBasicDataset(Dataset):
    def __init__(self, args, file_path, mode='mm'):
        self.args = args
        self.files, self.labels = read_data(file_path)
        self.feature_dir = self.args.feature_dir
        self.mode = mode
        self.mel_transform = get_mel_spectrogram_transform(args)

    def __getitem__(self, idx):
        # extract preprocessed feature
        audio_file, video_file, label = self.files[idx][0], self.files[idx][1], self.labels[idx]
        # fname, label = self.files[idx][0], self.labels[idx]
        # cur_dir = os.path.basename(os.path.dirname(fname))
        # cur_dir = cur_dir.replace('_audio_clips', '').replace('_video_clips', '')
        # clip_id = int(os.path.basename(fname).split('_')[1])
        # data_path = os.path.join(os.path.join(self.feature_dir, cur_dir), f'clip_{clip_id}.joblib')
        # spectrograms, video_tensor = read_feature(data_path)
        # construct return dict
        return_dict = {'label': self.labels[idx]}
        if self.mode == 'audio' or self.mode == 'mm':
            spectrograms = preprocess_audio(audio_file + '.wav', self.mel_transform, self.args)
            return_dict['audio_feature'] = spectrograms
        if self.mode == 'video' or self.mode == 'mm':
            video_tensor = preprocess_video(video_file + '.avi')
            return_dict['video_feature'] = video_tensor

        return return_dict

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    args = config.get_config()
    dset = VideoAudioBasicDataset(args, '../data/cross_validation/Fold_0_ValSet.txt')
    loader = iter(DataLoader(dset, batch_size=16, num_workers=0))

    data = next(loader)
    print(data)
