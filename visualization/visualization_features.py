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

# 特征配置字典 (只需配置差异项)
FEATURE_CONFIG = {
    'mar': {
        'title': 'Mouth Aspect Ratio (MAR)',
        'ylabel': 'MAR',
        'color': 'b',
        'legend': 'MAR',
        'linestyle': '-',
        'grid': True,
    },
    'ear_left': {
        'title': 'Left Eye Aspect Ratio (EAR)',
        'ylabel': 'EAR',
        'color': 'g',
        'legend': 'Left EAR',
        'linestyle': '-',
        'grid': True,
    },
    'ear_right': {
        'title': 'Right Eye Aspect Ratio (EAR)',
        'ylabel': 'EAR',
        'color': 'orange',
        'legend': 'Right EAR',
        'linestyle': '-',
        'grid': True,
    },
    
    # 'Face':{
    #     'title': 'Face Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'color': 'r',
    #     'legend': 'Face',
    # },
    # 'Left-arm': {
    #     'title': 'Left Arm Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'color': 'c',
    #     'legend': 'Left Arm',
    # },
    # 'Right-arm': {
    #     'title': 'Right Arm Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'color': 'm',
    #     'legend': 'Right Arm',
    # },
    # 'Left-leg': {
    #     'title': 'Left Leg Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'color': 'b',
    #     'legend': 'Left Leg',
    # },
    # 'Right-leg': {
    #     'title': 'Right Leg Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'color': 'g',
    #     'legend': 'Right Leg',
    # },
    # 'Torso-skin': {
    #     'title': 'Torso Skin Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'color': 'y',
    #     'legend': 'Torso Skin',
    # },
    # 'WholeFrameMotion': {
    #     'title': 'Whole Frame Motion Intensity',
    #     'ylabel': 'Intensity',
    #     'color': 'orange',
    #     'legend': 'WholeFrameMotion',
    # },
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

def get_valid_range(values):
    """自动计算合理的数值范围"""
    clean_values = clean_data(values)
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

def plot_feature(ax, time_sec, values, feature_name):
    """统一绘制特征曲线"""
    config = {**COMMON_CONFIG, **FEATURE_CONFIG[feature_name]}  # 合并配置
    
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
    ax.set_ylim(get_valid_range(values))  # 自动计算范围
    
    # 设置X轴格式
    time_labels = [sec_to_hms(sec) for sec in time_sec]
    tick_positions = np.linspace(0, len(time_sec)-1, 5, dtype=int)
    ax.set_xticks(time_sec[tick_positions])
    ax.set_xticklabels([time_labels[i] for i in tick_positions])
    
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

def visualize_features(features_json_path, output_img_path=None, features_to_plot=None, 
                      start_time=None, end_time=None, label_path=None):
    """主可视化函数
    Args:
        features_json_path: 输入JSON文件路径
        output_img_path: 输出图片路径(可选)
        features_to_plot: 要绘制的特征列表(可选)
        start_time: 开始时间(HH:MM:SS格式)(可选)
        end_time: 结束时间(HH:MM:SS格式)(可选)
        label_path: 标签文件路径(可选), 如果为None则自动从视频路径生成
    """
    # 加载数据
    with open(features_json_path) as f:
        data = json.load(f)
    
    features = data['features']
    fps = data['video_info']['fps']
    video_path = data['video_info']['path']
    
    # 自动生成标签文件路径
    if label_path is None:
        label_path = get_label_path_from_video_path(video_path)
    
    # 准备时间数据
    frames = [f['frame'] for f in features]
    time_sec = np.array(frames) / fps
    
    # 应用时间范围筛选
    if start_time or end_time:
        start_sec = hms_to_sec(start_time) if start_time else 0
        end_sec = hms_to_sec(end_time) if end_time else time_sec[-1]
        
        # 筛选数据
        mask = (time_sec >= start_sec) & (time_sec <= end_sec)
        features = [f for f, m in zip(features, mask) if m]
        time_sec = time_sec[mask]
        
        print(f"\n筛选时间段: {sec_to_hms(start_sec)} - {sec_to_hms(end_sec)}")
        print(f"筛选后帧数: {len(features)}")

    # 打印统计信息
    missing_stats = count_missing_values(features)
    print("\n空值统计结果:")
    print(f"总帧数: {missing_stats['total_frames']}")
    for feature in FEATURE_CONFIG:
        count = missing_stats[f'{feature}_missing']
        print(f"{feature.upper()}缺失: {count} ({count/missing_stats['total_frames']:.1%})")
    print(f"全部特征缺失: {missing_stats['all_missing']} ({missing_stats['all_missing']/missing_stats['total_frames']:.1%})")
    
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
    
    # 绘制选定的特征
    for idx, feature in enumerate(features_to_plot):
        values = [f.get(feature) for f in features]
        plot_feature(axes[idx], time_sec, values, feature)
    
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
    # 替换为您的实际路径
    # input_json = "face_landmarks_features.json"
    # output_img = "img/visualization_features.jpg"
    # ljh-baby-m zay-baby-f wqq-baby-f drz-m zzy-baby-m llj-baby-f
    # prefix = "/data/Leo/mm/data/raw_data/NanfangHospital/cry/drz-m/features/"
    # input_jsons = glob.glob(prefix + "*cam1_motion_features.json")

    # prefix = "/data/Leo/mm/data/raw_data/"
    # files = [
    #     # "NanfangHospital/cry/drz-m",
    #     # "ShenzhenUniversityGeneralHospital/non-cry/zlj-baby-f",
    #     "NanfangHospital/cry/llj-baby-f",
    #     # "NanfangHospital/non-cry/lyz-m",
    # ]
    # dirs = [prefix + file + "/features/*1_motion_features.json" for file in files]
    # input_jsons = [glob.glob(d) for d in dirs]
    # input_jsons = [item for sublist in input_jsons for item in sublist]

    # prefix = '/data/Leo/mm/data/Newborn200'
    # files = [
    #     '/Cry/features/48cm2.66kg',
    #     '/Cry/features/buduiqi_50cm3.4kg2',
    #     '/NoCry/features/50cm3.18kg2'
    # ]
    # input_jsons = [prefix + file + '_motion_features.json' for file in files]

    prefix = '/data/Leo/mm/data/Newborn200/'
    files = [
        "Cry/features/_49cm2.88kg_face_landmarks_features.json",
    ]
    input_jsons = [prefix + file for file in files]
    
    for input_json in input_jsons:
        print(f"\n处理文件: {input_json}")
        # output_img = "img/" + input_json.split('/')[-1].replace('_motion_features.json', '') + "_visualization.jpg"
        # label_path = input_json.replace('_motion_features.json', '.txt').replace('features', 'seg_result')
        
        output_img = 'image.png'
        visualize_features(input_json, output_img)
        # visualize_features(input_json, output_img, start_time="00:00:14", end_time="00:00:50")
        # visualize_features(input_json, output_img, start_time="00:04:30", end_time="00:05:00")
        # visualize_features(input_json, output_img, label_path=label_path)

    

    # 示例1: 绘制指定时间段的所有特征和标签 (最常用)
    # visualize_features(input_json, output_img, start_time="00:01:16", end_time="00:01:32")
    
    # 示例2: 自动绘制完整视频的所有特征和标签 (次常用)
    # visualize_features(input_json, output_img)
    
    # 示例3: 手动指定标签路径 (备用)
    # visualize_features(input_json, output_img, label_path="custom_labels.txt")
    
    # 示例4: 只绘制特定特征和标签 (备用)
    # visualize_features(input_json, output_img, features_to_plot=['mar', 'ear_left'])