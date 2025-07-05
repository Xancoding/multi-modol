import torch.nn as nn
import torch

class ChannelAttention1D(nn.Module):
    """轻量级通道注意力模块"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class CNN_FeatureAttn(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.2):
        super().__init__()
        # 卷积层部分
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 新增注意力模块
        self.attn = ChannelAttention1D(32)
        
        # 全连接层
        self.flatten_size = 32 * (input_size // 4)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 应用通道注意力
        x = self.attn(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)