#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:late_fusion.py
# author:xm
# datetime:2024/4/25 11:13
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch.nn as nn
import torch

from models.audio_models import audio_model_dict
from models.video_models import video_model_dict
import torch.nn.functional as F


class LateFusion(nn.Module):
    def __init__(self, audio_backbone: nn.Module, video_backbone: nn.Module, class_num=2, proj_dim=128):
        super(LateFusion, self).__init__()
        self.audio_backbone = audio_backbone
        self.video_backbone = video_backbone

        # self.audio_proj = nn.Sequential(
        #     nn.Linear(self.audio_backbone.embedding_dim, self.audio_backbone.embedding_dim // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.audio_backbone.embedding_dim // 4, proj_dim),
        # )
        #
        # self.video_proj = nn.Sequential(
        #     nn.Linear(self.video_backbone.embedding_dim, self.video_backbone.embedding_dim // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.video_backbone.embedding_dim // 4, proj_dim),
        # )

        # self.fuse_proj = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(proj_dim * 2, proj_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(proj_dim, class_num)
        # )

    def forward(self, audio_input, video_input):
        audio_out = self.audio_backbone(audio_input)
        video_out = self.video_backbone(video_input)
        audio_feats, video_feats = audio_out['feat'], video_out['feat']
        audio_logits, video_logits = audio_out['logits'], video_out['logits']

        # audio_proj_feats, video_proj_feats = self.audio_proj(audio_feats), self.video_proj(video_feats)
        # cat_feats = torch.cat((audio_proj_feats, video_proj_feats), dim=-1)
        # logits = self.fuse_proj(cat_feats)
        logits = (F.softmax(audio_logits, dim=-1) + F.softmax(video_logits, dim=-1)) / 2
        return {'logits': logits, 'audio_feats': audio_feats, 'video_feats': video_feats}


if __name__ == '__main__':
    audio_backbone = audio_model_dict['res18'](class_num=2)
    video_backbone = video_model_dict['slowfast_18'](class_num=2, fps=15)
    model = LateFusion(audio_backbone, video_backbone)
    audio_input = torch.randn((1, 1, 64, 192))
    video_input = torch.randn((1, 3, 45, 224, 224))
    out = model(audio_input, video_input)
    print(out['logits'])
