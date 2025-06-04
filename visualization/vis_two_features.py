import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import timedelta
import glob

# 公共配置
COMMON_CONFIG = {
    'linewidth': 1,
    'linestyle': '-',
    'grid': True,
    'grid_style': {'linestyle': '--', 'alpha': 0.5}
}

# 高对比度颜色方案 (dataset1亮色，dataset2深色)
COLOR_SCHEMES = {
    'dataset1': {  # 第一个数据集 - 全部使用高对比亮色
        'Face': '#FF0000',       # 纯红
        'Left-arm': '#FF0000',   # 纯红
        'Right-arm': '#FF0000',  # 纯红
        'Left-leg': '#FF0000',   # 纯红
        'Right-leg': '#FF0000',  # 纯红
        'WholeFrameMotion': '#FF0000' # 纯红
    },
    'dataset2': {  # 第二个数据集 - 全部使用高对比深色
        'Face': '#0000FF',       # 纯蓝
        'Left-arm': '#0000FF',   # 纯蓝
        'Right-arm': '#0000FF',  # 纯蓝
        'Left-leg': '#0000FF',   # 纯蓝
        'Right-leg': '#0000FF',  # 纯蓝
        'WholeFrameMotion': '#0000FF' # 纯蓝
    }
}

# 特征配置字典 (只需配置差异项)
FEATURE_CONFIG = {
    'Face':{
        'title': 'Face Motion Intensity',
        'ylabel': 'Intensity',
        'legend': 'Face',
    },
    # 'Left-arm': {
    #     'title': 'Left Arm Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'legend': 'Left Arm',
    # },
    # 'Right-arm': {
    #     'title': 'Right Arm Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'legend': 'Right Arm',
    # },
    # 'Left-leg': {
    #     'title': 'Left Leg Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'legend': 'Left Leg',
    # },
    # 'Right-leg': {
    #     'title': 'Right Leg Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'legend': 'Right Leg',
    # },
    'WholeFrameMotion': {
        'title': 'Whole Frame Motion Intensity',
        'ylabel': 'Intensity',
        'legend': 'WholeFrameMotion',
    },
}

def sec_to_hms(seconds):
    """将秒数转换为HH:MM:SS格式"""
    return str(timedelta(seconds=seconds)).split('.')[0]

def hms_to_sec(time_str):
    """将HH:MM:SS格式转换为秒数"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def get_label_path_from_video_path(video_path):
    """从视频路径生成标签文件路径"""
    # 移除文件扩展名
    base_path = os.path.splitext(video_path)[0]
    base_path = base_path.split('_cam')[0]
    # 添加_label.txt后缀
    return f"{base_path}_label.txt"

def read_label_file(label_path):
    """读取标签文件，返回时间段和标签列表"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = int(parts[2])
                    labels.append((start, end, label))
    else:
        print(f"\n警告: 标签文件 {label_path} 不存在")
    return labels

def count_missing_values(features):
    """自动统计各特征的空值数量"""
    stats = {'total_frames': len(features)}
    
    # 初始化计数器
    for feature in FEATURE_CONFIG:
        stats[f'{feature}_missing'] = 0
    stats['all_missing'] = 0
    
    for f in features:
        all_missing = True
        for feature in FEATURE_CONFIG:
            is_missing = f.get(feature) is None or np.isnan(f.get(feature))
            stats[f'{feature}_missing'] += int(is_missing)
            all_missing = all_missing and is_missing
        stats['all_missing'] += int(all_missing)
    
    return stats

def clean_data(values):
    """清洗数据，移除无效值"""
    return [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]

def get_valid_range(values1, values2):
    """自动计算合理的数值范围，考虑两个数据集"""
    clean_values1 = clean_data(values1)
    clean_values2 = clean_data(values2)
    clean_values = clean_values1 + clean_values2
    
    if not clean_values:
        return (0, 1)  # 默认范围
    
    max_val = max(clean_values)
    min_val = min(clean_values)
    
    # 自动调整范围，留出20%的余量
    range_padding = (max_val - min_val) * 0.2 if max_val != min_val else max_val * 0.2
    return (
        max(0, min_val - range_padding),  # 确保最小值不小于0
        max_val + range_padding
    )

def plot_feature(ax, time_sec, values, feature_name, dataset='dataset1'):
    """统一绘制特征曲线"""
    config = {**COMMON_CONFIG, **FEATURE_CONFIG[feature_name]}  # 合并配置
    
    # 根据数据集选择颜色
    color = COLOR_SCHEMES[dataset][feature_name]
    config['color'] = color
    config['legend'] = f"{config['legend']} ({dataset})"
    
    # 绘制曲线
    ax.plot(time_sec, 
            [v if v is not None and not np.isnan(v) else None for v in values],
            color=config['color'],
            linestyle=config['linestyle'],
            label=config['legend'],
            linewidth=config['linewidth'])
    
    # 设置图表属性
    ax.set_title(config['title'])
    ax.set_ylabel(config['ylabel'])
    
    # 添加网格和标签
    if config['grid']:
        ax.grid(**config['grid_style'])
    ax.legend()

def plot_label(ax, time_sec, labels):
    """绘制更美观的标签区域"""
    # 创建与时间轴对应的标签数组
    label_array = np.zeros_like(time_sec, dtype=int)
    
    # 填充标签数组
    for start, end, label in labels:
        mask = (time_sec >= start) & (time_sec <= end)
        label_array[mask] = label
    
    # 设置更美观的颜色和透明度
    crying_color = '#FF6B6B'  # 柔和的红色
    not_crying_color = '#6BCB77'  # 柔和的绿色
    alpha = 0.4  # 更柔和的透明度
    
    # 绘制标签区域 - 使用更精细的填充方式
    ax.fill_between(time_sec, 0, 1, 
                   where=(label_array==1),
                   facecolor=crying_color, 
                   edgecolor=crying_color,
                   alpha=alpha, 
                   label='Crying',
                   linewidth=0.5)
    
    ax.fill_between(time_sec, 0, 1,
                   where=(label_array==0),
                   facecolor=not_crying_color,
                   edgecolor=not_crying_color,
                   alpha=alpha,
                   label='Not Crying',
                   linewidth=0.5)
    
    # 设置图表属性 - 更专业的样式
    ax.set_title('Crying Detection Label', fontsize=10, pad=10)
    ax.set_ylabel('Status', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Not Crying', 'Crying'], fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    
    # 设置X轴格式 - 更精细的控制
    time_labels = [sec_to_hms(sec) for sec in time_sec]
    tick_positions = np.linspace(0, len(time_sec)-1, min(5, len(time_sec)), dtype=int)
    ax.set_xticks(time_sec[tick_positions])
    ax.set_xticklabels([time_labels[i] for i in tick_positions], fontsize=8)
    
    # 添加更专业的网格和图例
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    
    # 更美观的图例
    legend = ax.legend(loc='upper right', 
                      frameon=True, 
                      framealpha=0.8,
                      edgecolor='#333333',
                      facecolor='white',
                      fontsize=8)
    
    # 设置边框更美观
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#DDDDDD')
        spine.set_linewidth(0.8)
    
    # 添加水平分隔线
    ax.axhline(y=0.5, color='#DDDDDD', linestyle='-', linewidth=0.8, alpha=0.5)

def visualize_two_features(json_path1, json_path2, output_img_path=None, features_to_plot=None, 
                         start_time=None, end_time=None, label_path=None):
    """主可视化函数 - 处理两个JSON文件
    Args:
        json_path1: 第一个输入JSON文件路径
        json_path2: 第二个输入JSON文件路径
        output_img_path: 输出图片路径(可选)
        features_to_plot: 要绘制的特征列表(可选)
        start_time: 开始时间(HH:MM:SS格式)(可选)
        end_time: 结束时间(HH:MM:SS格式)(可选)
        label_path: 标签文件路径(可选), 如果为None则自动从第一个视频路径生成
    """
    # 加载第一个数据集
    with open(json_path1) as f:
        data1 = json.load(f)
    features1 = data1['features']
    fps1 = data1['video_info']['fps']
    video_path1 = data1['video_info']['path']
    
    # 加载第二个数据集
    with open(json_path2) as f:
        data2 = json.load(f)
    features2 = data2['features']
    fps2 = data2['video_info']['fps']
    video_path2 = data2['video_info']['path']
    
    # 检查FPS是否一致
    if fps1 != fps2:
        print(f"警告: 两个视频的FPS不一致 ({fps1} vs {fps2})")
    
    # 使用第一个视频的FPS
    fps = fps1
    
    # 自动生成标签文件路径
    if label_path is None:
        label_path = get_label_path_from_video_path(video_path1)
    
    # 准备时间数据
    frames1 = [f['frame'] for f in features1]
    time_sec1 = np.array(frames1) / fps
    
    frames2 = [f['frame'] for f in features2]
    time_sec2 = np.array(frames2) / fps
    
    # 检查时间轴长度是否一致
    if len(time_sec1) != len(time_sec2):
        print(f"警告: 两个视频的帧数不一致 ({len(time_sec1)} vs {len(time_sec2)})")
        min_frames = min(len(time_sec1), len(time_sec2))
        time_sec1 = time_sec1[:min_frames]
        time_sec2 = time_sec2[:min_frames]
        features1 = features1[:min_frames]
        features2 = features2[:min_frames]
    
    # 使用第一个时间轴
    time_sec = time_sec1
    
    # 应用时间范围筛选
    if start_time or end_time:
        start_sec = hms_to_sec(start_time) if start_time else 0
        end_sec = hms_to_sec(end_time) if end_time else time_sec[-1]
        
        # 筛选数据
        mask = (time_sec >= start_sec) & (time_sec <= end_sec)
        features1 = [f for f, m in zip(features1, mask) if m]
        features2 = [f for f, m in zip(features2, mask) if m]
        time_sec = time_sec[mask]
        
        print(f"\n筛选时间段: {sec_to_hms(start_sec)} - {sec_to_hms(end_sec)}")
        print(f"筛选后帧数: {len(features1)}")
    
    # 打印统计信息
    print("\n第一个数据集空值统计结果:")
    missing_stats1 = count_missing_values(features1)
    print(f"总帧数: {missing_stats1['total_frames']}")
    for feature in FEATURE_CONFIG:
        count = missing_stats1[f'{feature}_missing']
        print(f"{feature.upper()}缺失: {count} ({count/missing_stats1['total_frames']:.1%})")
    
    print("\n第二个数据集空值统计结果:")
    missing_stats2 = count_missing_values(features2)
    print(f"总帧数: {missing_stats2['total_frames']}")
    for feature in FEATURE_CONFIG:
        count = missing_stats2[f'{feature}_missing']
        print(f"{feature.upper()}缺失: {count} ({count/missing_stats2['total_frames']:.1%})")
    
    # 确定绘制哪些特征
    features_to_plot = features_to_plot or list(FEATURE_CONFIG.keys())
    features_to_plot = [f for f in features_to_plot if f in FEATURE_CONFIG]
    
    # 读取标签数据
    labels = read_label_file(label_path)
    if labels:
        print("\n成功加载标签数据")
        print(f"标签时间段数量: {len(labels)}")
    
    # 创建画布 (增加一个子图用于标签)
    n_subplots = len(features_to_plot) + (1 if labels else 0)
    fig, axes = plt.subplots(n_subplots, 1, 
                           figsize=(12, 3*n_subplots))
    if n_subplots == 1: 
        axes = [axes]
    
    # 绘制选定的特征 - 两个数据集
    for idx, feature in enumerate(features_to_plot):
        values1 = [f.get(feature) for f in features1]
        values2 = [f.get(feature) for f in features2]
        
        # 先绘制两个数据集
        plot_feature(axes[idx], time_sec, values1, feature, dataset='dataset1')
        plot_feature(axes[idx], time_sec, values2, feature, dataset='dataset2')
        
        # 设置统一的Y轴范围
        axes[idx].set_ylim(get_valid_range(values1, values2))
        
        # 设置X轴格式
        time_labels = [sec_to_hms(sec) for sec in time_sec]
        tick_positions = np.linspace(0, len(time_sec)-1, 5, dtype=int)
        axes[idx].set_xticks(time_sec[tick_positions])
        axes[idx].set_xticklabels([time_labels[i] for i in tick_positions])
    
    # 绘制标签
    if labels:
        plot_label(axes[-1], time_sec, labels)
    
    # 调整布局并保存/显示
    plt.tight_layout()
    if output_img_path:
        plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_img_path}")
    else:
        plt.show()

# 使用示例
if __name__ == "__main__":
    prefix = "/data/Leo/mm/data/raw_data/"
    files = [
        "NanfangHospital/cry/drz-m",
        # "ShenzhenUniversityGeneralHospital/non-cry/zlj-baby-f",
        # "NanfangHospital/cry/llj-baby-f",
        # "NanfangHospital/non-cry/lyz-m",
    ]
    dirs = [prefix + file + "/features/*_motion_features.json" for file in files]
    input_jsons = [glob.glob(d) for d in dirs]
    input_jsons = [item for sublist in input_jsons for item in sublist]
    
    json1 = input_jsons[0]
    json2 = input_jsons[1]
    output_img = "comparison_visualization.jpg"
    
    visualize_two_features(json1, json2, output_img)