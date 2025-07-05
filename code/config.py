n_splits = 10
seed = 42
# Set data
audioSampleRate = 16000
slidingWindows = 2.5
step = 1.5
# slidingWindows = 3.0
# step = 1.5
# slidingWindows = 2.5
# step = 2.5


# Feature Extraction
FFTwindow = 256
FFTOverlap = 128
MFCCFiliterNum = 26

# 神经网络训练配置
batch_size = 64
learning_rate = 0.001
weight_decay = 0.01
max_epochs = 100
# max_epochs = 1
patience = 5  # 早停等待轮数

model_type = 'cnn' # 模型类型：'lgbm', 'rf', 'svm'
hard_cases_num = 15  # 难例数量
easy_cases_num = 15  # 易例数量