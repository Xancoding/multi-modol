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
from features.Feature_Extraction_Audio import acoustic_features_and_spectrogram, NICUWav2Segments, extract_raw_acoustic_features
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
    acoustic_feature, acoustic_feature_name = acoustic_features_and_spectrogram(sample["data"])
    return acoustic_feature, sample["label"], acoustic_feature_name

def extract_body_motion_features(dataDir):
    """Extract body motion features"""
    file = utils.get_motion_feature_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    motion_feature, motion_feature_name, _ = body_features(file, config.slidingWindows, config.step, labelDir)
    return motion_feature, motion_feature_name

def extract_facial_features(dataDir):
    """Extract facial features"""
    file = utils.get_face_landmark_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    facial_feature, facial_feature_name, _ = facial_features(file, config.slidingWindows, config.step, labelDir)
    return facial_feature, facial_feature_name

def load_or_extract_features(wav_files):
    """加载或提取特征数据（包含名称）"""
    dir = os.path.dirname(wav_files[0])
    feature_dir = os.path.join(os.path.dirname(dir), 'Features')
    os.makedirs(feature_dir, exist_ok=True)
    subject_data = {}
    
    for file in tqdm(wav_files, desc="处理文件中"):
        feature_path = os.path.join(feature_dir, utils.generate_feature_filename(file))
        
        # 尝试加载已保存的特征（包含名称）
        if os.path.exists(feature_path):
            try:
                data = np.load(feature_path, allow_pickle=True)
                subject_data.setdefault(utils.extract_subject_id(file), []).append({
                    'acoustic': data['acoustic'],
                    'motion': data['motion'],
                    'face': data['face'],
                    'label': data['label'],
                    'acoustic_feature_names': data['acoustic_feature_names'].tolist(),
                    'motion_feature_names': data['motion_feature_names'].tolist(),
                    'face_feature_names': data['face_feature_names'].tolist(),
                })
                continue
            except Exception as e:
                print(f"加载特征失败，将重新生成: {e}")

        # 提取新特征（确保名称被捕获）
        acoustic_feat, label, audio_names = extract_audio_features(file)
        motion_feat, motion_names = extract_body_motion_features(file)
        face_feat, face_names = extract_facial_features(file)       

        np.savez(
            feature_path,
            acoustic=acoustic_feat,
            motion=motion_feat,
            face=face_feat,
            label=label,
            acoustic_feature_names=audio_names,
            motion_feature_names=motion_names,
            face_feature_names=face_names,
        )

        subject_data.setdefault(utils.extract_subject_id(file), []).append({
            'acoustic': acoustic_feat,
            'motion': motion_feat,
            'face': face_feat,
            'label': label,
            'acoustic_feature_names': audio_names,
            'motion_feature_names': motion_names,
            'face_feature_names': face_names,
        })
    
    return subject_data

def prepare_feature_matrices(subject_data):
    """Prepare feature matrices and print statistics"""
    print("\nPreparing feature matrices...")
    all_acoustic = []
    all_motion = []
    all_face = []
    all_labels = []
    all_subject_ids = []

    # 获取特征名称（假设所有样本的特征名称相同）
    first_sample = next(iter(subject_data.values()))[0]
    acoustic_feature_names = first_sample.get('acoustic_feature_names', None)
    motion_feature_names = first_sample.get('motion_feature_names', None)
    face_feature_names = first_sample.get('face_feature_names', None)

    if None in [acoustic_feature_names, motion_feature_names, face_feature_names]:
        raise ValueError("特征名称未正确加载！请检查特征提取函数") 
    
    # 统计婴儿哭闹情况（基于基本ID）
    crying_infants = set()
    non_crying_infants = set()
    infant_status = {}  # 记录每个基本ID是否有哭泣
    
    # 统计片段哭闹情况
    crying_segments = 0
    non_crying_segments = 0
    
    for subj_id_with_timestamp, samples in tqdm(subject_data.items(), desc="Processing subjects"):
        # 提取基本婴儿ID（去掉时间戳部分）
        base_subj_id = subj_id_with_timestamp.split('_')[0]
        
        # 初始化婴儿状态（如果尚未记录）
        if base_subj_id not in infant_status:
            infant_status[base_subj_id] = False
        
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
                
                # 添加各模态特征（保持原始ID）
                all_acoustic.append(sample['acoustic'][i])
                all_motion.append(sample['motion'][i])
                all_face.append(sample['face'][i])
                all_labels.append(label)
                all_subject_ids.append(subj_id_with_timestamp)
        
        # 更新婴儿状态
        if has_crying:
            infant_status[base_subj_id] = True
    
    # 根据基本ID统计婴儿数量
    for base_subj_id, has_cried in infant_status.items():
        if has_cried:
            crying_infants.add(base_subj_id)
        else:
            non_crying_infants.add(base_subj_id)
    
    # 打印统计信息
    print("\n=== 数据统计 ===")
    print(f"婴儿总数: {len(infant_status)}")  # 基于基本ID的数量
    print(f"哭泣的婴儿数量: {len(crying_infants)}")
    print(f"未哭泣的婴儿数量: {len(non_crying_infants)}")
    print(f"\n片段总数: {len(all_labels)}")
    print(f"带哭声的片段数量: {crying_segments}")
    print(f"不带哭声的片段数量: {non_crying_segments}")
    print(f"哭闹片段占比: {crying_segments/len(all_labels):.2%}")
    
    # 打印特征维度信息
    print("\n=== 特征维度 ===")
    print(f"音频特征维度: {len(acoustic_feature_names)}")
    print(f"运动特征维度: {len(motion_feature_names)}")
    print(f"面部特征维度: {len(face_feature_names)}")
    
    return (
        all_subject_ids, 
        all_acoustic, 
        all_motion, 
        all_face, 
        all_labels,
        acoustic_feature_names,
        motion_feature_names,
        face_feature_names,
    )

def load_data():
    """Load and prepare all data"""
    prefix = config.dataDir
    ori_wav_files = glob.glob(f"{prefix}*.wav")
    excluded_wav_files = [prefix + x + '.wav' for x in 
                        [
                            'hbxd-m_2025-07-29-14-45-40',
                            'hbxd-m_2025-07-29-15-06-59',
                            'ydw-baby-f_2025-07-29-13-39-36',
                            'ysqd-f_2025-07-29-15-32-40',
                            'whq-baby-m_2025-07-29-11-48-29',
                            'lj-baby-m_2025-07-29-12-06-14',
                            'mxt-baby-f_2025-07-29-13-09-45',
                            'ydw-baby-f_2025-07-29-15-17-50',
                            'ysqd-f_2025-07-29-16-34-26',
                            'cjyd-m_2025-08-05-15-43-31',
                            'dxh-baby-f_2025-08-05-13-48-59',
                            'hyy-baby-m_2025-08-05-13-29-37',
                            'thz-m_2025-08-05-12-45-55',
                            'hmx-baby-f_2025-08-11-17-21-54',
                            'wss-baby-f_2025-08-11-17-34-58',
                            'hmx-baby-f_2025-08-14-16-54-55',

                            'fm-baby-m_2025-08-05-17-05-20',

                            # 'hym-baby-f_2025-07-29-13-02-20',
                            # 'xgx-baby-m_2025-07-29-13-29-54',
                            # 'hym-baby-f_2025-07-29-15-26-27',
                            # 'cjyx-f_2025-07-29-16-48-32',
                            # 'lxt-f_2025-07-29-17-14-18',
                            # 'xgx-baby-m_2025-07-29-15-39-53'
                            # 'lyj-baby-f_2025-08-02-15-54-55',
                            # 'cjq-baby-m_2025-08-02-17-12-24',
                            # 'lwc-m_2025-08-02-16-47-18'
                            # 'wxl-baby-m_2025-08-05-12-19-40',  
                            # 'cfy-baby-m_2025-08-05-14-05-54',  
                            # 'hj-baby-m_2025-08-07-16-28-22',
                            # 'zl-baby-m_2025-08-18-17-25-39'
                        ]]    
    wav_files = [f for f in ori_wav_files if f not in excluded_wav_files]

    subject_data = load_or_extract_features(wav_files)
    print(f"Total subjects: {len(subject_data)}")
    
    (subject_ids, acoustic_features, motion_features, 
     face_features, labels, acoustic_feature_names,
     motion_feature_names, face_feature_names) = prepare_feature_matrices(subject_data)
    
    print(f"\nFeature dimensions:")
    print(f"Audio features: {np.array(acoustic_features).shape} (names: {len(acoustic_feature_names)})")
    print(f"Motion features: {np.array(motion_features).shape} (names: {len(motion_feature_names)})")
    print(f"Face features: {np.array(face_features).shape} (names: {len(face_feature_names)})")
    
    return (
        subject_ids,
        (acoustic_features, motion_features, face_features,),  
        labels,
        (acoustic_feature_names, motion_feature_names, face_feature_names)  
    )