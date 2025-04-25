#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:data_utils.py
# author:xm
# datetime:2024/4/25 13:04
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os

from torch.utils.data import DataLoader

from dataset.basic_dataset import VideoAudioBasicDataset
from dataset.utils.preprocessing_utils import read_feature


def get_dataset(args, fold, mode):
    dataset_dict = {}
    cross_val_dir = args.cross_val_dir
    # train dataset
    train_dataset = VideoAudioBasicDataset(args, os.path.join(cross_val_dir, f'Fold_{fold}_TrainSet.txt'), mode)
    dataset_dict['train'] = train_dataset

    # val dataset
    val_dataset = VideoAudioBasicDataset(args, os.path.join(cross_val_dir, f'Fold_{fold}_ValSet.txt'), mode)
    dataset_dict['val'] = val_dataset

    # test dataset
    test_dataset = VideoAudioBasicDataset(args, os.path.join(cross_val_dir, f'Fold_{fold}_TestSet.txt'), mode)
    dataset_dict['test'] = test_dataset

    return dataset_dict


def get_data_loader(args, dataset_dict):
    dataloader_dict = {}
    # train_loader
    train_dataset = dataset_dict['train']
    dataloader_dict['train'] = DataLoader(train_dataset, args.train_batch_size, shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=False)

    # val_loader
    val_dataset = dataset_dict['val']
    val_loader = DataLoader(val_dataset, args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    dataloader_dict['val'] = val_loader

    # test_loader
    test_dataset = dataset_dict['test']
    test_loader = DataLoader(test_dataset, args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=False)
    dataloader_dict['test'] = test_loader

    return dataloader_dict
