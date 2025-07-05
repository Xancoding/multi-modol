import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.2):
        super(CNN, self).__init__()
        
        # 卷积层部分
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 计算卷积后的特征维度
        # 假设输入形状为 (batch_size, 1, input_size)
        # 经过两次池化(kernel_size=2)，特征长度变为 input_size // 4
        self.flatten_size = 32 * (input_size // 4)
        
        # 全连接层部分
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 输入形状: (batch_size, input_size)
        # 添加通道维度: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # 卷积部分
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接部分
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)