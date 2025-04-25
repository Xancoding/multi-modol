#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:__init__.py.py
# author:xm
# datetime:2024/4/25 11:03
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from models.audio_models.resnet.resnet import make_res18_network

audio_model_dict = {
    'res18': make_res18_network,
}
