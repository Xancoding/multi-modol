#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:multi_modal_pipeline.py
# author:xm
# datetime:2024/4/25 15:01
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import time

from accelerate import Accelerator
import torch

from models.audio_models import audio_model_dict
from models.mm_models import get_fusion_model
from models.pipelines.basepipeline import BasePipeline
from models.video_models import video_model_dict


class MonoModalPipeline(BasePipeline):
    def __init__(self, args, fold: int, accelerator: Accelerator, logger):
        super(MonoModalPipeline, self).__init__(args, fold, accelerator, logger)

    def set_model(self):
        if self.mode == 'audio':
            backbone = audio_model_dict[self.args.audio_model](class_num=2, use_fc=True)
        elif self.mode == 'video':
            backbone = video_model_dict[self.args.video_model](class_num=2, fps=15, use_fc=True)
        else:
            return NotImplementedError, "only support audio and video modality"
        return backbone

    def train_one_epoch(self, epoch):
        self.model.train()
        # batch iteration
        avg_loss = 0

        for i, data in enumerate(self.dataloader_dict['train']):
            x,  label = data[f'{self.mode}_feature'], data['label']
            # forward
            logits = self.model(x)['logits']

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
                x,  label = data[f'{self.mode}_feature'], data['label']
                logits = self.model(x)['logits']
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

