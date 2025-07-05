# pytorch_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np
import config

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    """将PyTorch模型包装成sklearn兼容的格式"""
    def __init__(self, model_class, epochs=config.max_epochs, batch_size=config.batch_size, 
                 learning_rate=config.learning_rate, device='cuda' if torch.cuda.is_available() else 'cpu',
                 class_weight=None, random_state=config.seed):
        self.model_class = model_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.class_weight = class_weight
        self.random_state = random_state
        
        # 设置随机种子
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
    def fit(self, X, y):
        # 检查输入数据
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X.copy()).to(self.device)  # 添加.copy()
        y_tensor = torch.LongTensor(y.copy()).to(self.device)  # 添加.copy()  
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化模型
        input_size = X.shape[1]
        num_classes = len(self.classes_)
        self.model_ = self.model_class(input_size, num_classes).to(self.device)
        
        # 定义损失函数（支持类别权重）
        if self.class_weight is not None:
            weight = torch.FloatTensor(self.class_weight).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()
            
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # 训练循环
        self.model_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            self.model_.eval()
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            self.model_.eval()
            outputs = self.model_(X_tensor)
            return torch.softmax(outputs, dim=1).cpu().numpy()