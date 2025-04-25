#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:late_fusion.py
# author:xm
# datetime:2024/4/25 11:13
# software: PyCharm

"""
Late fusion model supporting audio and dual video streams (cam0 and cam1)
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.audio_models import audio_model_dict
from models.video_models import video_model_dict


class LateFusion(nn.Module):
    def __init__(self, 
                 audio_backbone: nn.Module, 
                 video_backbone_cam0: nn.Module,
                 video_backbone_cam1: nn.Module,
                 args,
                 class_num=2, 
                 proj_dim=128):
        super(LateFusion, self).__init__()
        self.audio_backbone = audio_backbone
        self.video_backbone_cam0 = video_backbone_cam0
        self.video_backbone_cam1 = video_backbone_cam1
        self.mm_config = args.mm_config
        
        # Optional projection layers (commented out by default)
        # self.audio_proj = nn.Sequential(
        #     nn.Linear(self.audio_backbone.embedding_dim, proj_dim),
        #     nn.ReLU(inplace=True)
        # )
        # 
        # self.video_proj_cam0 = nn.Sequential(
        #     nn.Linear(self.video_backbone_cam0.embedding_dim, proj_dim),
        #     nn.ReLU(inplace=True)
        # )
        # 
        # self.video_proj_cam1 = nn.Sequential(
        #     nn.Linear(self.video_backbone_cam1.embedding_dim, proj_dim),
        #     nn.ReLU(inplace=True)
        # )
        # 
        # self.fuse_proj = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(proj_dim * 3, proj_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(proj_dim, class_num)
        # )

    def forward(self, **inputs):
        # 简化的模态处理
        audio_logits = self.audio_backbone(inputs['audio'])['logits'] if 'audio' in inputs else None
        cam0_logits = self.video_backbone_cam0(inputs['video_cam0'])['logits'] if 'video_cam0' in inputs else None
        cam1_logits = self.video_backbone_cam1(inputs['video_cam1'])['logits'] if 'video_cam1' in inputs else None

        # 动态计算有效模态
        active_logits = [F.softmax(x, dim=-1) for x in [audio_logits, cam0_logits, cam1_logits] if x is not None]
        
        # 平均融合
        logits = sum(active_logits) / len(active_logits) if active_logits else None
        
        return {'logits': logits}

if __name__ == '__main__':
    # Test with dual video streams
    audio_backbone = audio_model_dict['res18'](class_num=2)
    video_backbone = video_model_dict['slowfast_18'](class_num=2, fps=15)
    
    # Create model with two video backbones (could use different architectures)
    model = LateFusion(
        audio_backbone,
        video_backbone,  # cam0 backbone
        video_backbone,  # cam1 backbone (could be different)
        class_num=2
    )
    
    # Create test inputs
    audio_input = torch.randn((1, 1, 64, 192))
    video_input_cam0 = torch.randn((1, 3, 45, 224, 224))
    video_input_cam1 = torch.randn((1, 3, 45, 224, 224))
    
    # Forward pass
    out = model(audio_input, video_input_cam0, video_input_cam1)
    print("Output logits shape:", out['logits'].shape)
    print("Sample output:", out['logits'])
