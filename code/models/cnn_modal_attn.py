import torch.nn as nn
import torch

class ModalAttention(nn.Module):
    """模态注意力模块"""
    def __init__(self, num_modals=3, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_modals, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modals),
            nn.Softmax(dim=1)
        )
    
    def forward(self, feats):
        # feats: [b, num_modals, feat_dim]
        weights = self.fc(feats.mean(dim=2))  # 全局平均后计算权重
        return (feats * weights.unsqueeze(2)).sum(dim=1)  # 加权求和

class CNN_ModalAttn(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.2):
        super().__init__()
        # 三个模态共享的CNN特征提取器
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.motion_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.face_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 模态注意力
        self.modal_attn = ModalAttention()
        
        # 全连接层
        self.flatten_size = 32 * (input_size // 4)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, audio, motion, face):
        # 各模态特征提取
        audio_feat = self.audio_cnn(audio.unsqueeze(1))
        motion_feat = self.motion_cnn(motion.unsqueeze(1))
        face_feat = self.face_cnn(face.unsqueeze(1))
        
        # 拼接模态特征 [b, 3, feat_dim]
        combined = torch.stack([
            audio_feat.view(audio.size(0), -1),
            motion_feat.view(motion.size(0), -1),
            face_feat.view(face.size(0), -1)
        ], dim=1)
        
        # 模态注意力融合
        fused = self.modal_attn(combined)
        
        # 分类
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        return self.fc2(x)