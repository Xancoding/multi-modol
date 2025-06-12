# Standard library imports
import os
import glob

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Local imports
from Feature_Extraction_Audio import acoustic_features_and_spectrogram, NICUWav2Segments
from Feature_Extraction_Body import body_features
from Feature_Extraction_Face import facial_features
import config
import utils

# Global time mapping table
TIME_MAPPING = None

def create_classifier(model_type, random_state=None):
    """Create specified classifier"""
    if model_type == 'svm':
        return SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=random_state, probability=True)
    elif model_type == 'fnn':
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    elif model_type == 'knn':
        return KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    elif model_type == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    elif model_type == 'xgb':
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state, scale_pos_weight=1)
    elif model_type == 'lgbm':
        return LGBMClassifier(random_state=random_state, class_weight='balanced', n_estimators=100, verbosity=-1)
    raise ValueError(f"Unknown model type: {model_type}")

def standardize_features(X_train, X_test=None):
    """Standardize feature data"""
    scaler = StandardScaler().fit(X_train)
    if X_test is not None:
        return scaler.transform(X_train), scaler.transform(X_test)
    return scaler.transform(X_train)

def calculate_classification_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score']
    }

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

def balance_dataset_by_subject(subject_data):
    """Balance dataset (by subject)"""
    print("\nBalancing dataset (segment level)...")
    
    all_samples = []
    for subj_id, samples in subject_data.items():
        for sample in samples:
            n_frames = len(sample['label'])
            for i in range(n_frames):
                all_samples.append({
                    'subject_id': subj_id,
                    'acoustic': sample['acoustic'][i],
                    'motion': sample['motion'][i],
                    'face': sample['face'][i],
                    'label': sample['label'][i]
                })
    
    df = pd.DataFrame(all_samples)
    pos_samples = df[df['label'] == 1]
    neg_samples = df[df['label'] == 0]
    
    balanced_neg = resample(
        neg_samples,
        replace=False,
        n_samples=len(pos_samples),
        random_state=config.seed
    )
    
    balanced_df = pd.concat([pos_samples, balanced_neg])
    
    balanced_data = {}
    for subj_id in balanced_df['subject_id'].unique():
        subj_samples = balanced_df[balanced_df['subject_id'] == subj_id]
        balanced_data[subj_id] = [{
            'acoustic': subj_samples['acoustic'].values,
            'motion': subj_samples['motion'].values,
            'face': subj_samples['face'].values,
            'label': subj_samples['label'].values
        }]
    
    return balanced_data

def prepare_feature_matrices(subject_data):
    """Prepare feature matrices"""
    print("\nPreparing feature matrices...")
    all_acoustic = []
    all_motion = []
    all_face = []
    all_labels = []
    all_subject_ids = []
    
    for subj_id, samples in tqdm(subject_data.items(), desc="Processing subjects"):
        for sample in samples:
            n_frames = len(sample['label'])
            for i in range(n_frames):
                all_acoustic.append(sample['acoustic'][i])
                all_motion.append(sample['motion'][i])
                all_face.append(sample['face'][i])
                all_labels.append(sample['label'][i])
                all_subject_ids.append(subj_id)
    
    return all_subject_ids, all_acoustic, all_motion, all_face, all_labels

def evaluate_classifier_performance(features, labels, subject_ids, model_name, model_type='svm'):
    """Evaluate classifier performance"""
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    
    X = np.concatenate([np.array(f) for f in features], axis=1) if isinstance(features, tuple) else np.array(features)
    y = np.array(labels)
    groups = np.array(subject_ids)
    
    sgkf = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    
    for _, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        X_train, X_test = standardize_features(X_train, X_test)
        model = create_classifier(model_type, config.seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics.update(calculate_classification_metrics(y_test, y_pred))
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"\n{model_name} ({model_type.upper()}) results:")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1-score: {avg_metrics['f1-score']:.4f}")    
    
    return avg_metrics

def evaluate_single_modality_performance(features, labels, modality_name, subject_ids, model_type='svm'):
    """Evaluate single modality performance"""
    print(f"\n=== Evaluating {modality_name} features ({model_type.upper()}) ===")
    return evaluate_classifier_performance(features, labels, subject_ids, f"{modality_name} model", model_type)

def evaluate_multimodal_voting_performance(acoustic_features, motion_features, face_features, labels, subject_ids, model_type='svm'):
    """Evaluate multimodal voting performance"""
    print(f"\n=== Evaluating multimodal voting fusion ({model_type.upper()}) ===")
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    
    X_audio = np.array(acoustic_features)
    X_motion = np.array(motion_features)
    X_face = np.array(face_features)
    y = np.array(labels).astype(int)
    groups = np.array(subject_ids)
    
    sgkf = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    
    for _, (train_idx, test_idx) in enumerate(sgkf.split(X_audio, y, groups)):
        X_train_audio, X_test_audio = X_audio[train_idx], X_audio[test_idx]
        X_train_motion, X_test_motion = X_motion[train_idx], X_motion[test_idx]
        X_train_face, X_test_face = X_face[train_idx], X_face[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scalers = {
            'audio': StandardScaler().fit(X_train_audio),
            'motion': StandardScaler().fit(X_train_motion),
            'face': StandardScaler().fit(X_train_face)
        }
        X_test_audio = scalers['audio'].transform(X_test_audio)
        X_test_motion = scalers['motion'].transform(X_test_motion)
        X_test_face = scalers['face'].transform(X_test_face)
        
        models = {
            'audio': create_classifier(model_type, config.seed),
            'motion': create_classifier(model_type, config.seed),
            'face': create_classifier(model_type, config.seed)
        }
        models['audio'].fit(scalers['audio'].transform(X_train_audio), y_train)
        models['motion'].fit(scalers['motion'].transform(X_train_motion), y_train)
        models['face'].fit(scalers['face'].transform(X_train_face), y_train)
        
        # Soft voting
        weights = {'audio': 0.4, 'motion': 0.3, 'face': 0.3}
        y_pred = (
            weights['audio'] * models['audio'].predict_proba(X_test_audio)[:, 1] +
            weights['motion'] * models['motion'].predict_proba(X_test_motion)[:, 1] +
            weights['face'] * models['face'].predict_proba(X_test_face)[:, 1]
        )
        y_pred = (y_pred >= 0.5).astype(int)

        metrics.update(calculate_classification_metrics(y_test, y_pred))
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"\nMultimodal voting fusion ({model_type.upper()}) results:")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1-score: {avg_metrics['f1-score']:.4f}")

    return avg_metrics

def run_experiment():
    """Main experiment function"""
    utils.initialize_random_seed(config.seed)
    
    prefix = '/data/Leo/mm/data/Newborn200/data/'
    wav_files = glob.glob(f"{prefix}*.wav")
    print(f"Found {len(wav_files)} audio files")

    subject_data = load_or_extract_features(wav_files)
    print(f"Total subjects: {len(subject_data)}")
    
    balanced_data = balance_dataset_by_subject(subject_data)
    subject_ids, acoustic_features, motion_features, face_features, labels = prepare_feature_matrices(balanced_data)

    print(f"\nFeature dimensions:")
    print(f"Audio features: {np.array(acoustic_features).shape}")
    print(f"Motion features: {np.array(motion_features).shape}")
    print(f"Face features: {np.array(face_features).shape}")
    
    model_type = 'svm'  # Options: 'svm', 'fnn', 'knn', 'rf', 'xgb', 'lgbm'
    
    evaluate_single_modality_performance(acoustic_features, labels, "Audio", subject_ids, model_type)
    evaluate_single_modality_performance(motion_features, labels, "Motion", subject_ids, model_type)
    evaluate_single_modality_performance(face_features, labels, "Face", subject_ids, model_type)
    evaluate_single_modality_performance((acoustic_features, motion_features, face_features), labels, "Multimodal", subject_ids, model_type)
    evaluate_multimodal_voting_performance(acoustic_features, motion_features, face_features, labels, subject_ids, model_type)
    print("\n=== All evaluations completed ===")

if __name__ == "__main__":
    run_experiment()