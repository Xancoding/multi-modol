# 导入必要的库
import matplotlib.pyplot as plt
FONT_SIZE = 20
FEATURE_LEN = 15

# 定义特征名称和重要性的字典
feature_importance = {
    'mar_min': 3.0,
    'left_ear_min': 2.6,
    'mfcc_6_mean': 2.4,
    'face_median': 2.0,
    'right_ear_median': 1.9,
    'chroma_6_std': 1.8,
    'right_ear_min': 1.6,
    'left_arm_median': 1.6,
    'mar_median': 1.5,
    'mar_mean': 1.5,
    'chroma_11_std': 1.5,
    'chroma_8_std': 1.4,
    'mar_max': 1.4,
    'right_arm_median': 1.3,
    'zcr_std': 1.3,
    'mfcc_7_median': 1.2,
    'left_arm_std': 1.2,
    'left_ear_median': 1.2,
    'energy_entropy_median': 1.1,
    'mfcc_2_std': 1.1
}
feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:FEATURE_LEN])  # 截取前FEATURE_LEN个特征
# 提取键和值以创建柱状图
features = list(feature_importance.keys())
importance_values = list(feature_importance.values())

# 创建柱状图
plt.figure(figsize=(14, 8))  # 设置图表大小，便于显示长标签
colors = [
    '#fde725',  # 1st (Highest Importance - Lightest Yellow)
    '#e4e138',  # 2nd
    '#c8de41',  # 3rd
    '#addc58',  # 4th
    '#94d871',  # 5th
    '#7cd387',  # 6th
    '#67ce9c',  # 7th
    '#53c8af',  # 8th
    '#41c1bf',  # 9th
    '#32b5c7',  # 10th
    '#25a7c9',  # 11th
    '#1f98c8',  # 12th
    '#2186c2',  # 13th
    '#2373b5',  # 14th
    '#245f9e'   # 15th (Lowest Importance - Darkest Blue)
]
bars = plt.bar(features, importance_values, color=colors[:len(features)])  # 使用渐变颜色

# 在每个柱状图上添加具体数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,  # x位置在柱子中心，y位置在柱子上方
             f'{height:.1f}', ha='center', va='bottom', fontsize=FONT_SIZE)  # 标签内容，水平居中，垂直底部对齐

# 设置标题和轴标签
plt.title('Multimodal Feature Ranking', fontsize=FONT_SIZE+4, fontweight='bold')
plt.xlabel('Feature Name', fontsize=FONT_SIZE, fontweight='bold')
plt.ylabel('Feature Importance', fontsize=FONT_SIZE, fontweight='bold')
plt.ylim(0, max(importance_values) * 1.1) 
# 旋转x轴标签以提高可读性
plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE-2)
plt.yticks(fontsize=FONT_SIZE-2) 

# 添加网格以便阅读
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局以避免标签被裁剪
plt.tight_layout()

plt.savefig('rank.png', dpi=300)