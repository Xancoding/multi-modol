# Standard library imports
import os
import glob

# Third-party imports
import numpy as np
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.data import Dataset

# Local imports
from features.Feature_Extraction_Audio import acoustic_features_and_spectrogram, NICUWav2Segments
from features.Feature_Extraction_Body import body_features
from features.Feature_Extraction_Face import facial_features
import config
import utils

class FeatureDataset(Dataset):
    """PyTorch Dataset for features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset"""
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    return torch.FloatTensor(sample_weights)

def extract_audio_features(dataDir):
    """Extract audio features"""
    labelDir = utils.get_label_file_path(dataDir)
    sample = NICUWav2Segments(dataDir, labelDir)
    acoustic_feature, _ = acoustic_features_and_spectrogram(sample["data"])
    return acoustic_feature, sample["label"]

def extract_body_motion_features(dataDir):
    """Extract body motion features"""
    file = utils.get_motion_feature_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    motion_feature, _, _ = body_features(file, config.slidingWindows, config.step, labelDir)
    return motion_feature

def extract_facial_features(dataDir):
    """Extract facial features"""
    file = utils.get_face_landmark_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    facial_feature, _, _ = facial_features(file, config.slidingWindows, config.step, labelDir)
    return facial_feature

def load_or_extract_features(wav_files):
    """加载或提取特征数据"""
    feature_dir = '/data/Leo/mm/data/Newborn200/Features'
    os.makedirs(feature_dir, exist_ok=True)
    subject_data = {}
    existing_files = set(os.listdir(feature_dir))
    
    print("\n正在加载/生成特征数据...")
    for file in tqdm(wav_files, desc="处理文件中"):
        label_path = utils.get_label_file_path(file)
        feature_file = utils.generate_feature_filename(label_path)
        feature_path = os.path.join(feature_dir, feature_file)
        
        if feature_file in existing_files:
            try:
                data = np.load(feature_path)
                subject_id = utils.extract_subject_id(file)
                if subject_id not in subject_data:
                    subject_data[subject_id] = []
                subject_data[subject_id].append({
                    'acoustic': data['acoustic'],
                    'motion': data['motion'],
                    'face': data['face'],
                    'label': data['label']
                })
                continue
            except Exception as e:
                print(f"\n加载特征文件 {feature_file} 失败，将重新生成: {str(e)}")
        
        subject_id = utils.extract_subject_id(file)
        acoustic_feat, label = extract_audio_features(file)
        motion_feat = extract_body_motion_features(file)
        face_feat = extract_facial_features(file)

        # 检查特征维度是否一致
        if len(acoustic_feat) != len(motion_feat) or len(acoustic_feat) != len(face_feat):
            print(f"\n警告: {subject_id} 的特征维度不一致!")
            print(f"声学特征维度: {len(acoustic_feat)}")
            print(f"运动特征维度: {len(motion_feat)}")
            print(f"面部特征维度: {len(face_feat)}")
            print(f"标签长度: {len(label)}")
            continue

        sample = {
            'acoustic': acoustic_feat,
            'motion': motion_feat,
            'face': face_feat,
            'label': label
        }
        np.savez(feature_path,
                acoustic=acoustic_feat,
                motion=motion_feat,
                face=face_feat,
                label=label)
        
        if subject_id not in subject_data:
            subject_data[subject_id] = []
        subject_data[subject_id].append(sample)
    
    return subject_data

def prepare_feature_matrices(subject_data):
    """Prepare feature matrices and print statistics"""
    print("\nPreparing feature matrices...")
    all_acoustic = []
    all_motion = []
    all_face = []
    all_labels = []
    all_subject_ids = []
    
    # 统计婴儿哭闹情况
    crying_infants = set()
    non_crying_infants = set()
    
    # 统计片段哭闹情况
    crying_segments = 0
    non_crying_segments = 0
    
    for subj_id, samples in tqdm(subject_data.items(), desc="Processing subjects"):
        has_crying = False
        for sample in samples:
            n_frames = len(sample['label'])
            for i in range(n_frames):
                label = sample['label'][i]
                if label == 1:  # 假设1表示哭闹
                    crying_segments += 1
                    has_crying = True
                else:
                    non_crying_segments += 1
                
                all_acoustic.append(sample['acoustic'][i])
                all_motion.append(sample['motion'][i])
                all_face.append(sample['face'][i])
                all_labels.append(label)
                all_subject_ids.append(subj_id)
        
        if has_crying:
            crying_infants.add(subj_id)
        else:
            non_crying_infants.add(subj_id)
    
    # 打印统计信息
    print("\n=== 数据统计 ===")
    print(f"婴儿总数: {len(subject_data)}")
    print(f"哭泣的婴儿数量: {len(crying_infants)}")
    print(f"未哭泣的婴儿数量: {len(non_crying_infants)}")
    print(f"\n片段总数: {len(all_labels)}")
    print(f"带哭声的片段数量: {crying_segments}")
    print(f"不带哭声的片段数量: {non_crying_segments}")
    print(f"哭闹片段占比: {crying_segments/len(all_labels):.2%}")
    
    return all_subject_ids, all_acoustic, all_motion, all_face, all_labels

def load_data():
    """Load and prepare all data"""
    prefix = '/data/Leo/mm/data/Newborn200/data/'
    wav_files = glob.glob(f"{prefix}*.wav")
    print(f"Found {len(wav_files)} audio files")
    
    subject_data = load_or_extract_features(wav_files)
    print(f"Total subjects: {len(subject_data)}")
    
    subject_ids, acoustic_features, motion_features, face_features, labels = prepare_feature_matrices(subject_data)
    
    print(f"\nFeature dimensions:")
    print(f"Audio features: {np.array(acoustic_features).shape}")
    print(f"Motion features: {np.array(motion_features).shape}")
    print(f"Face features: {np.array(face_features).shape}")
    
    return subject_ids, acoustic_features, motion_features, face_features, labels