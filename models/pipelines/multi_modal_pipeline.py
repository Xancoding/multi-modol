#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:multi_modal_pipeline.py
# author:xm
# datetime:2024/4/25 15:01
# software: PyCharm

"""
Multi-modal pipeline supporting audio and dual video streams (cam0 and cam1)
"""

import time
from accelerate import Accelerator
import torch
from models.audio_models import audio_model_dict
from models.mm_models import get_fusion_model
from models.pipelines.basepipeline import BasePipeline
from models.video_models import video_model_dict


class MultiModalPipeline(BasePipeline):
    def __init__(self, args, fold: int, accelerator: Accelerator, logger):
        super(MultiModalPipeline, self).__init__(args, fold, accelerator, logger)

    def set_model(self):
        audio_backbone = audio_model_dict[self.args.audio_model](class_num=2, use_fc=True)
        # Initialize two video backbones (can be same or different architectures)
        video_backbone_cam0 = video_model_dict[self.args.video_model](class_num=2, fps=15, use_fc=True)
        video_backbone_cam1 = video_model_dict[self.args.video_model](class_num=2, fps=15, use_fc=True)
        
        model = get_fusion_model[self.args.fusion_mode](
            audio_backbone, 
            video_backbone_cam0,
            video_backbone_cam1,
            self.args
        )
        return model

    def train_one_epoch(self, epoch):
        self.model.train()
        avg_loss = 0

        for i, data in enumerate(self.dataloader_dict['train']):
            # Get both video streams
            # audio_x = data['audio_feature']
            # video_x_cam0 = data['video_feature_cam0']
            # video_x_cam1 = data['video_feature_cam1']
            label = data['label']
            inputs = {}
            # 根据配置选择输入模态
            if self.args.mm_config in ['all', 'cam0', 'cam1']:
                inputs['audio'] = data['audio_feature']
                
            if self.args.mm_config in ['all', 'cam0']:
                inputs['video_cam0'] = data['video_feature_cam0']
                
            if self.args.mm_config in ['all', 'cam1']:
                inputs['video_cam1'] = data['video_feature_cam1']
            
            # Forward pass with both video streams
            logits = self.model(**inputs)['logits']

            loss = self.criterion(logits, label)
            avg_loss += loss.item()

            # params update
            self.update_params(loss)

            pred = torch.max(logits, dim=-1)[1]
            self.metrics_update(pred, label)

            self.it += 1

        train_metrics = self.metrics_computation()
        avg_loss = round(avg_loss / self.iter_per_epoch, 6)
        train_metrics['loss'] = avg_loss
        return train_metrics

    def evaluate(self, val_loader):
        self.model.eval()
        loss = 0
        with torch.inference_mode():
            for i, data in enumerate(val_loader):
                label = data['label']

                inputs = {}
                # 根据配置选择输入模态
                if self.args.mm_config in ['all', 'cam0', 'cam1']:
                    inputs['audio'] = data['audio_feature']
                if self.args.mm_config in ['all', 'cam0']:
                    inputs['video_cam0'] = data['video_feature_cam0']
                    
                if self.args.mm_config in ['all', 'cam1']:
                    inputs['video_cam1'] = data['video_feature_cam1']
                
                # Forward pass with both video streams
                logits = self.model(**inputs)['logits']
                                
                # gather all predictions and targets
                all_logits, all_labels = self.accelerator.gather_for_metrics((logits, label))
                loss += self.criterion(all_logits, all_labels).item()

                pred = torch.max(all_logits, dim=-1)[1]
                self.metrics_update(pred, all_labels)
            
            val_metrics = self.metrics_computation()
            val_metrics['loss'] = round(loss / len(val_loader), 6)

        self.metrics_reset()
        self.model.train()
        return val_metrics