seed = 42
audioSampleRate = 16000

# Data Split 
n_splits = 10
slidingWindows = 2.5
step = 1.5

# Feature Extraction
FFTwindow = 256
FFTOverlap = 128
MFCCFiliterNum = 26 

model_type = 'lgbm' # model type: 'svm', 'rf', 'lgbm'

dataDir = '/data/Leo/mm/data/NICU50/data/'
# dataDir = '/data/Leo/mm/data/NEWBORN200/data/'