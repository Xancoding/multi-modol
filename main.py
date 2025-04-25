#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:main.py
# author:xm
# datetime:2024/4/24 13:34
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os
import warnings

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from torch.backends import cudnn

from config import get_config
from dataset.utils.preprocessing_utils import split_data, clean_data, store_features
from models.pipelines.mono_modal_pipeline import MonoModalPipeline
from models.pipelines.multi_modal_pipeline import MultiModalPipeline
from utils.common_utils import set_logger

warnings.filterwarnings('ignore')

logger = get_logger(__name__, log_level='INFO')


def get_model_backbone(args):
    mode = args.mode
    model_backbone = mode
    if mode == 'mm':
        model_backbone += f'-{args.mm_config}-{args.fusion_mode}'
    if mode == 'audio' or mode == 'mm':
        model_backbone += f'-{args.audio_model}'
    if mode == 'video' or mode == 'mm':
        model_backbone += f'-{args.video_model}'

    return model_backbone


def main_worker(args, fold, accelerator_config):
    accelerator = Accelerator(log_with='wandb', project_config=accelerator_config,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    # wandb init
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=f'Infant-Cry-Detection-{model_backbone}-Fold-{fold}', config=args)
    if args.mode == 'mm':
        pipeline = MultiModalPipeline(args, fold, accelerator, logger)
    else:
        pipeline = MonoModalPipeline(args, fold, accelerator, logger)

    if fold == 0:
        logger.info(f'args: {args}')

    test_metrics = pipeline.train()
    accelerator.end_training()
    return test_metrics


if __name__ == '__main__':
    # args and basic settings (set seed to ensure reproducible, set logger to record training logs)
    args = get_config()
    # clean_data(args.data_dir)
    # split_data(args)
    # store_features(args)
   

    # wandb offline setting. use wandb sync log_path after training to upload logs to cloud
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_MODE"] = args.wandb_mode
    # NCCL setting for RTX 3090/4090 DDP
    os.environ["NCCL_P2P_DISABLE"] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'


    model_backbone = get_model_backbone(args)
    args.model_backbone = model_backbone

    set_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    set_logger(os.path.join(args.log_dir, f'{model_backbone}.log'))

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    accelerator_config = ProjectConfiguration(project_dir='.', logging_dir='./logs')

    start_fold = 3
    for fold in range(start_fold, args.k_fold):
        test_metrics = main_worker(args, fold, accelerator_config)
        for key in metrics.keys():
            metrics[key].append(test_metrics[key])

    for key in metrics.keys():
        v = torch.tensor(metrics[key])
        metrics[key] = {'mean': round(torch.mean(v, dim=-1).item(), 6), 'std': round(torch.std(v, dim=-1).item(), 6)}

    logger.info(f'{model_backbone}: {args.k_fold}-fold cross-validation, '
                f'{args.average}-average test metrics: {metrics}')
