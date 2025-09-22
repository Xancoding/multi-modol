seed = 42
audioSampleRate = 16000

# Data Split 
n_splits = 10
slidingWindows = 2.5
step = 1.5
hard_cases_num = 15  # 难例数量
easy_cases_num = 15  # 易例数量

# Feature Extraction
FFTwindow = 256
FFTOverlap = 128
MFCCFiliterNum = 26 

model_type = 'lgbm' # 模型类型：'svm', 'rf', 'lgbm'

dataDir = '/data/Leo/mm/data/NICU/data/'
# dataDir = '/data/Leo/mm/data/Newborn200/data/'
# dataDir = '/data/Leo/mm/data/ShenzhenUniversityGeneralHospital/data/'
# dataDir = '/data/Leo/mm/data/NanfangHospital/data/'