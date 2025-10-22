# violin.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
import warnings 
from PIL import Image
import os
import sys
from sklearn.preprocessing import MinMaxScaler

# 1. 获取当前脚本的绝对路径 (chart/violinplot.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 向上移动一级目录 (到达包含 chart 和 code 的父目录)
parent_dir = os.path.join(current_dir, '..')
# 3. 将 code 文件夹的路径添加到 Python 模块搜索路径中
sys.path.append(os.path.join(parent_dir, 'code'))
import data_loader

def _plot_single_violin(data, title, xlabel, savefile, y_min, y_max) -> None:
    if len(data) == 0:
        warnings.warn(f"Dataset for '{xlabel}' is empty. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 8))
    
    violin_parts = ax.violinplot(
        [data],
        widths=0.7,
        # showmeans=True,
        showmedians=True,
        showextrema=True,
        # showbox=True, # 报错
    )

    for pc in violin_parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('blue')
        pc.set_alpha(0.7)
    
        violin_parts['cmaxes'].set_color('red')
        violin_parts['cmins'].set_color('red')
        violin_parts['cmedians'].set_color('red')
        violin_parts['cmedians'].set_linewidth(3)
        # violin_parts['cmeans'].set_color('red')
        # violin_parts['cmeans'].set_linewidth(3)
    
    ax.set_xticks([1])
    ax.set_xticklabels([xlabel])
    
    y_range = y_max - y_min
    y_pad = y_range * 0.05
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.yaxis.grid(True)
    
    plt.savefig(savefile)
    plt.close(fig)

def _plot_combined_violin(file_0: str, file_1: str, combined_file: str) -> None:
    """读取两个图文件，左右拼接，并保存为新的文件。"""
    if not os.path.exists(file_0) or not os.path.exists(file_1):
        warnings.warn("Input files for combination do not exist. Skipping combination.")
        return

    try:
        img0 = Image.open(file_0)
        img1 = Image.open(file_1)

        width0, height0 = img0.size
        width1, height1 = img1.size
        
        combined_width = width0 + width1
        combined_height = max(height0, height1)
        
        new_img = Image.new('RGB', (combined_width, combined_height), 'white')
        
        new_img.paste(img0, (0, 0))
        new_img.paste(img1, (width0, 0))

        new_img.save(combined_file)
        
    except Exception as e:
        warnings.warn(f"Error during image combination: {e}")


def plot_violin(data_0: Union[List[float], np.ndarray], title_0: str, xlabel_0: str, savefile_0: str, 
                data_1: Union[List[float], np.ndarray], title_1: str, xlabel_1: str, savefile_1: str, 
                combined_file: str,
                ) -> None:
    """绘制两个独立的、Y轴对齐的小提琴图，并拼接成一个图。"""
    
    all_data = np.concatenate([data_0, data_1])
    if len(all_data) == 0:
        warnings.warn("Both datasets are empty. Cannot plot.")
        return
          
    y_min = np.min(all_data)
    y_max = np.max(all_data)
    
    # 绘制两个独立的图
    _plot_single_violin(data_0, title_0, xlabel_0, savefile_0, y_min, y_max)
    _plot_single_violin(data_1, title_1, xlabel_1, savefile_1, y_min, y_max)

    # 拼接图像
    _plot_combined_violin(savefile_0, savefile_1, combined_file)
    os.remove(savefile_0)
    os.remove(savefile_1)

if __name__ == '__main__':
    participant_ids, feature_sets, feature_names, labels, scenes = data_loader.load_data()   
    acoustic_features, motion_features, face_features = feature_sets
    acoustic_feature_names, motion_feature_names, face_feature_names = feature_names

    # features= acoustic_features
    # names = acoustic_feature_names
    # FEATURE_INDEX = 13 # face_median

    # features= motion_features
    # names = motion_feature_names
    # FEATURE_INDEX = 1 # face_median

    features = face_features
    names = face_feature_names
    # FEATURE_INDEX = 8  # left_ear_min
    FEATURE_INDEX = 3  # mar_min

    feat = np.array(features)
    y = np.array(labels)

    scaler = MinMaxScaler()
    normalized_feat = scaler.fit_transform(feat)
    
    FEATURE_NAME = names[FEATURE_INDEX]
    selected_normalized_feature = normalized_feat[:, FEATURE_INDEX]
    data_label_0 = selected_normalized_feature[y == 0]
    data_label_1 = selected_normalized_feature[y == 1]

    plot_violin(
        data_0=data_label_0, 
        title_0=f"{FEATURE_NAME} Distribution (NoCry)", 
        xlabel_0="NoCry",
        savefile_0=f'png/violin_{FEATURE_NAME}_nocry.png',
        
        data_1=data_label_1, 
        title_1=f"{FEATURE_NAME} Distribution (Cry)", 
        xlabel_1="Cry",
        savefile_1=f'png/violin_{FEATURE_NAME}_cry.png',

        combined_file=f'png/violin_{FEATURE_NAME}.png',
    )