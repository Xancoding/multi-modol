#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:__init__.py.py
# author:xm
# datetime:2024/4/24 15:48
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from models.video_models.slowfast.slowfast import slowfast_18, slowfast_50

video_model_dict = {
    'slowfast_18': slowfast_18,
    'slowfast_50': slowfast_50,
}
