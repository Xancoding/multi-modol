#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:common_utils.py
# author:xm
# datetime:2024/4/23 14:53
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os
import logging


def set_logger(log_path):
    """
    :param log_path: where to save log file
    :return:
    """
    ensure_dir(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Since every call to the set_logger function creates a handler, which can cause duplicate printing problems,
    # it is necessary to determine if the handler is already present in the root logger
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def ensure_dir(file_path):
    fname, ext = os.path.splitext(file_path)
    if ext:
        # extract file dir
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    # check if dir is existing, else create it
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
