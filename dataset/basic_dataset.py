#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:basic_dataset.py
# datetime:2024/4/23 11:10
# software: PyCharm

"""
Multi-modal dataset supporting audio and dual video streams (cam0 and cam1)
"""

from typing import Dict
from torch.utils.data import Dataset, DataLoader
import os
import config
from dataset.utils.preprocessing_utils import (
    read_data, 
    get_mel_spectrogram_transform,
    preprocess_audio,
    preprocess_video,
    read_feature
)

class VideoAudioBasicDataset(Dataset):
    def __init__(self, args, file_path: str, mode: str = 'mm'):
        """
        Args:
            args: Configuration arguments
            file_path: Path to data file containing audio/video paths and labels
            mode: Operation mode ('audio', 'video', or 'mm' for multi-modal)
        """
        self.args = args
        self.mm_config = args.mm_config 
        # Read data now expects format: audio_path, video_cam0_path, video_cam1_path, label
        self.files, self.labels = read_data(file_path)
        self.feature_dir = self.args.feature_dir
        self.mode = mode
        self.mel_transform = get_mel_spectrogram_transform(args)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing features based on the specified mode.
        For dual video support, we now process both cam0 and cam1 streams.
        """
        # Unpack all file paths
        audio_file, video_file_cam0, video_file_cam1 = self.files[idx]

        
        return_dict = {'label': self.labels[idx]}
        
        # 处理音频
        if self.mode == 'audio' or (self.mode == 'mm' and self.mm_config in ['all', 'cam0', 'cam1']):
            spectrograms = preprocess_audio(audio_file + '.wav', self.mel_transform, self.args)
            return_dict['audio_feature'] = spectrograms
            
        # 处理视频
        if self.mode == 'video' or self.mode == 'mm':
            if self.mm_config in ['all', 'cam0']:
                video_tensor_cam0 = preprocess_video(video_file_cam0 + '.avi')
                return_dict['video_feature_cam0'] = video_tensor_cam0
                
            if self.mm_config in ['all', 'cam1']:
                video_tensor_cam1 = preprocess_video(video_file_cam1 + '.avi')
                return_dict['video_feature_cam1'] = video_tensor_cam1

        return return_dict

    def __len__(self) -> int:
        return len(self.labels)


if __name__ == '__main__':
    args = config.get_config()
    # Test with dual video support
    dset = VideoAudioBasicDataset(args, '../data/cross_validation/Fold_0_ValSet.txt')
    loader = iter(DataLoader(dset, batch_size=16, num_workers=0))

    data = next(loader)
    print("Batch data structure:")
    print(f"Audio features shape: {data['audio_feature'].shape}")
    print(f"Video cam0 features shape: {data['video_feature_cam0'].shape}")
    print(f"Video cam1 features shape: {data['video_feature_cam1'].shape}")
    print(f"Labels: {data['label']}")