import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttentionModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dim: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # --- 特征注意力机制 (深化部分) ---
        # 使用 MLP 来生成注意力分数，可以学习更复杂的非线性关系
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # 批量归一化
            nn.ReLU(),                  # 激活函数
            nn.Dropout(dropout_rate),   # Dropout 防止过拟合
            nn.Linear(hidden_dim, input_dim) # 输出与 input_dim 相同维度的分数
        )

        # --- 分类头 ---
        # 考虑到注意力机制可能已经学习了特征的重要性，分类头可以保持简单
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状: (batch_size, input_dim)

        # --- 生成注意力权重 ---
        # 通过 MLP 生成注意力原始分数
        attention_scores = self.attention_mlp(x)
        # 应用 Sigmoid 函数，将注意力权重限制在 0 到 1 之间
        attention_weights = torch.sigmoid(attention_scores)

        # --- 应用注意力并加入残差连接 ---
        # 原始特征乘以注意力权重
        attended_features = x * attention_weights
        # 添加残差连接：加权后的特征 + 原始特征
        # 这有助于保留原始信息并稳定训练
        fused_features = attended_features + x # 简单的残差连接

        # --- 分类 ---
        output = self.classifier(fused_features)
        return output