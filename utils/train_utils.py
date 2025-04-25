#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:train_utils.py
# author:xm
# datetime:2024/4/25 12:36
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
from transformers import get_cosine_schedule_with_warmup


def build_optimizer_and_scheduler(args, model, t_total):
    model_param = model.parameters()
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model_param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model_param, lr=args.lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    # warm_up_with_cosine_lr
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total),
                                                num_training_steps=t_total)

    return optimizer, scheduler
