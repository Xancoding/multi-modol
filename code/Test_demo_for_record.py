# Standard library imports
import os
import glob

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

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
from Preprocessing import NICUWav2Segments
from Feature_Extraction_Audio import acoustic_features_and_spectrogram
from Feature_Extraction_Body import body_features
from Feature_Extraction_Face import facial_features
import config
import utils

# 全局时间映射表
TIME_MAPPING = None

# ====================== 提取的独立函数 ======================

def create_classifier(model_type, random_state=None):
    """创建指定类型的分类器"""
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
    raise ValueError(f"未知的模型类型: {model_type}")

def standardize_features(X_train, X_test=None):
    """标准化特征数据"""
    scaler = StandardScaler().fit(X_train)
    if X_test is not None:
        return scaler.transform(X_train), scaler.transform(X_test)
    return scaler.transform(X_train)

def calculate_classification_metrics(y_true, y_pred):
    """计算分类评估指标"""
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score']
    }

def record_misclassified_samples(y_true, y_pred, subject_ids, original_indices, fold):
    """记录错误分类的样本"""
    misclassified = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            misclassified.append({
                'subject_id': subject_ids[i],
                'original_index': original_indices[i],
                'true_label': y_true[i],
                'predicted_label': y_pred[i],
                'fold': fold
            })
    return misclassified

def create_multimodal_classifiers(model_type):
    """创建多模态分类器"""
    return {
        'audio': create_classifier(model_type),
        'motion': create_classifier(model_type),
        'face': create_classifier(model_type)
    }

# ====================== 主功能函数 ======================

def extract_audio_features(dataDir):
    """提取音频特征"""
    labelDir = utils.get_label_file_path(dataDir)
    sample = NICUWav2Segments(dataDir, labelDir)
    acoustic_feature, feature_names = acoustic_features_and_spectrogram(sample["data"])
    return acoustic_feature, sample["label"]

def extract_body_motion_features(dataDir):
    """提取身体运动特征"""
    file = utils.get_motion_feature_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    motion_feature, feature_names, _ = body_features(file, config.slidingWindows, config.step, labelDir)
    return motion_feature

def extract_facial_features(dataDir):
    """提取面部特征"""
    file = utils.get_face_landmark_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    facial_feature, feature_names, _ = facial_features(file, config.slidingWindows, config.step, labelDir)
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
    """平衡数据集（按受试者）"""
    print("\n正在平衡数据集（片段级别）...")
    
    all_samples = []
    for subj_id, samples in subject_data.items():
        for sample in samples:
            n_frames = len(sample['label'])
            for i in range(n_frames):
                original_start = i * config.step
                all_samples.append({
                    'subject_id': subj_id,
                    'acoustic': sample['acoustic'][i],
                    'motion': sample['motion'][i],
                    'face': sample['face'][i],
                    'label': sample['label'][i],
                    'original_start': original_start,
                    'original_index': i
                })
    
    df = pd.DataFrame(all_samples)
    pos_samples = df[df['label'] == 1]
    neg_samples = df[df['label'] == 0]
    
    unique_subjects_before = df.groupby('subject_id')['label'].max().reset_index()
    cry_babies_before = len(unique_subjects_before[unique_subjects_before['label'] == 1])
    non_cry_babies_before = len(unique_subjects_before[unique_subjects_before['label'] == 0])
    
    print(f"\n原始数据:")
    print(f"总片段={len(df)} (哭泣片段:{len(pos_samples)} | 非哭泣片段:{len(neg_samples)})")
    print(f"总婴儿={len(unique_subjects_before)} (哭泣婴儿:{cry_babies_before} | 非哭泣婴儿:{non_cry_babies_before})")

    balanced_neg = resample(
        neg_samples,
        replace=False,
        n_samples=len(pos_samples),
        random_state=config.seed
    )
    
    balanced_df = pd.concat([pos_samples, balanced_neg])
    
    print(f"\n平衡后数据:")
    print(f"总片段={len(balanced_df)} (哭泣片段:{len(pos_samples)} | 非哭泣片段:{len(balanced_neg)})")
    unique_subjects_after = balanced_df.groupby('subject_id')['label'].max().reset_index()
    cry_babies_after = len(unique_subjects_after[unique_subjects_after['label'] == 1])
    non_cry_babies_after = len(unique_subjects_after[unique_subjects_after['label'] == 0])
    print(f"总婴儿={len(unique_subjects_after)} (哭泣婴儿:{cry_babies_after} | 非哭泣婴儿:{non_cry_babies_after})")
    
    balanced_data = {}
    for subj_id in balanced_df['subject_id'].unique():
        subj_samples = balanced_df[balanced_df['subject_id'] == subj_id]
        balanced_data[subj_id] = [{
            'acoustic': subj_samples['acoustic'].values,
            'motion': subj_samples['motion'].values,
            'face': subj_samples['face'].values,
            'label': subj_samples['label'].values,
            'original_start': subj_samples['original_start'].values,
            'original_index': subj_samples['original_index'].values
        }]
    
    return balanced_data

def prepare_feature_matrices(subject_data):
    """准备特征矩阵"""
    print("\n正在准备特征矩阵...")
    all_acoustic = []
    all_motion = []
    all_face = []
    all_labels = []
    all_subject_ids = []
    all_original_indices = []
    time_mapping = []
    
    global_index = 0
    for subj_id, samples in tqdm(subject_data.items(), desc="处理受试者"):
        for sample in samples:
            n_frames = len(sample['label'])
            for i in range(n_frames):
                all_acoustic.append(sample['acoustic'][i])
                all_motion.append(sample['motion'][i])
                all_face.append(sample['face'][i])
                all_labels.append(sample['label'][i])
                all_subject_ids.append(subj_id)
                all_original_indices.append(global_index)
                
                time_mapping.append({
                    'subject_id': subj_id,
                    'balanced_index': global_index,
                    'original_start': sample['original_start'][i],
                    'original_index': sample['original_index'][i],
                    'window_size': config.slidingWindows,
                    'step': config.step
                })
                global_index += 1
    
    global TIME_MAPPING
    TIME_MAPPING = pd.DataFrame(time_mapping)
    return all_subject_ids, all_acoustic, all_motion, all_face, all_labels, all_original_indices

def locate_original_time_by_index(original_index):
    """根据索引定位原始时间"""
    record = TIME_MAPPING[TIME_MAPPING['balanced_index'] == original_index].iloc[0]
    start = record['original_start']
    end = start + record['window_size']
    return {
        'subject_id': record['subject_id'],
        'time_range': (start, end),
        'original_index': record['original_index'],
        'window_size': record['window_size'],
        'step': record['step']
    }

def analyze_misclassified_samples(error_df):
    """分析错误分类样本"""
    results = []
    for _, row in error_df.iterrows():
        time_info = locate_original_time_by_index(row['original_index'])
        results.append({
            'subject_id': row['subject_id'],
            'balanced_index': row['original_index'],
            'original_index': time_info['original_index'],
            'start_time': time_info['time_range'][0],
            'end_time': time_info['time_range'][1],
            'true_label': row['true_label'],
            'predicted_label': row['predicted_label'],
            'fold': row['fold']
        })
    return pd.DataFrame(results)

def evaluate_classifier_performance(features, labels, subject_ids, original_indices, model_name, model_type='svm'):
    """评估分类器性能"""
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    misclassified_samples = []
    
    X = np.concatenate([np.array(f) for f in features], axis=1) if isinstance(features, tuple) else np.array(features)
    y = np.array(labels)
    groups = np.array(subject_ids)
    
    sgkf = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subjects = groups[test_idx]
        test_original_indices = [original_indices[i] for i in test_idx]
        
        X_train, X_test = standardize_features(X_train, X_test)
        model = create_classifier(model_type, config.seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        misclassified_samples.extend(record_misclassified_samples(
            y_test, y_pred, test_subjects, test_original_indices, fold_idx + 1
        ))
        
        metrics.update(calculate_classification_metrics(y_test, y_pred))
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"\n{model_name} ({model_type.upper()}) 评估结果:")
    print(f"准确率: {avg_metrics['accuracy']:.4f}")
    print(f"精确率: {avg_metrics['precision']:.4f}")
    print(f"召回率: {avg_metrics['recall']:.4f}")
    print(f"F1分数: {avg_metrics['f1-score']:.4f}")    
    
    return avg_metrics

def evaluate_single_modality_performance(features, labels, modality_name, subject_ids, original_indices, model_type='svm'):
    """评估单模态性能"""
    print(f"\n=== 正在评估 {modality_name} 特征 ({model_type.upper()}) ===")
    return evaluate_classifier_performance(features, labels, subject_ids, original_indices, f"{modality_name}模型", model_type)

def evaluate_multimodal_voting_performance(acoustic_features, motion_features, face_features, labels, subject_ids, original_indices, model_type='svm'):
    """评估多模态投票性能"""
    print(f"\n=== 正在评估基于投票的多模态融合 ({model_type.upper()}) ===")
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    misclassified_samples = []
    
    X_audio = np.array(acoustic_features)
    X_motion = np.array(motion_features)
    X_face = np.array(face_features)
    y = np.array(labels).astype(int)
    groups = np.array(subject_ids)
    
    sgkf = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    cv_iter = sgkf.split(X_audio, y, groups)
    
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(cv_iter, total=config.n_splits, desc="多模态投票CV")):
        X_train_audio, X_test_audio = X_audio[train_idx], X_audio[test_idx]
        X_train_motion, X_test_motion = X_motion[train_idx], X_motion[test_idx]
        X_train_face, X_test_face = X_face[train_idx], X_face[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subjects = groups[test_idx]
        test_original_indices = [original_indices[i] for i in test_idx]
        
        scalers = {
            'audio': StandardScaler().fit(X_train_audio),
            'motion': StandardScaler().fit(X_train_motion),
            'face': StandardScaler().fit(X_train_face)
        }
        X_test_audio = scalers['audio'].transform(X_test_audio)
        X_test_motion = scalers['motion'].transform(X_test_motion)
        X_test_face = scalers['face'].transform(X_test_face)
        
        models = create_multimodal_classifiers(model_type)
        models['audio'].fit(scalers['audio'].transform(X_train_audio), y_train)
        models['motion'].fit(scalers['motion'].transform(X_train_motion), y_train)
        models['face'].fit(scalers['face'].transform(X_train_face), y_train)
        
        # 软投票
        weights = {'audio': 0.4, 'motion': 0.3, 'face': 0.3}
        y_pred = (
            weights['audio'] * models['audio'].predict_proba(X_test_audio)[:, 1] +
            weights['motion'] * models['motion'].predict_proba(X_test_motion)[:, 1] +
            weights['face'] * models['face'].predict_proba(X_test_face)[:, 1]
        )
        y_pred = (y_pred >= 0.5).astype(int)

        misclassified_samples.extend(record_misclassified_samples(
            y_test, y_pred, test_subjects, test_original_indices, fold_idx + 1
        ))
        
        metrics.update(calculate_classification_metrics(y_test, y_pred))
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"\n多模态投票融合 ({model_type.upper()}) 结果:")
    print(f"准确率: {avg_metrics['accuracy']:.4f}")
    print(f"精确率: {avg_metrics['precision']:.4f}")
    print(f"召回率: {avg_metrics['recall']:.4f}")
    print(f"F1分数: {avg_metrics['f1-score']:.4f}")

    return avg_metrics

def run_experiment(data_scope="all", target_subjects=None):
    """运行实验主函数"""
    utils.initialize_random_seed(config.seed)
    
    prefix = '/data/Leo/mm/data/Newborn200/data/'
    wav_files = glob.glob(f"{prefix}*.wav")
    
    if data_scope == "specific":
        if not target_subjects:
            raise ValueError("target_subjects不能为空当data_scope='specific'")
        wav_files = [f for f in wav_files if utils.extract_subject_id(f) in target_subjects]
        print("=== 指定婴儿实验（保持平衡）===")
    else:
        print("=== 婴儿哭声检测多模态评估（全部数据）===")
    
    print(f"找到 {len(wav_files)} 个音频文件")

    subject_data = load_or_extract_features(wav_files)
    print(f"共 {len(subject_data)} 个受试者")
    
    balanced_data = balance_dataset_by_subject(subject_data)
    
    subject_ids, acoustic_features, motion_features, face_features, labels, original_indices = prepare_feature_matrices(balanced_data)

    print(f"\n特征维度统计:")
    print(f"声学特征维度: {np.array(acoustic_features).shape}")
    print(f"运动特征维度: {np.array(motion_features).shape}")
    print(f"面部特征维度: {np.array(face_features).shape}")
    
    model_type = 'svm'  # 可选 'svm', 'fnn', 'knn', 'rf', 'xgb', 'lgbm'
    
    evaluate_single_modality_performance(acoustic_features, labels, "声学特征", subject_ids, original_indices, model_type)
    evaluate_single_modality_performance(motion_features, labels, "运动特征", subject_ids, original_indices, model_type)
    evaluate_single_modality_performance(face_features, labels, "面部特征", subject_ids, original_indices, model_type)
    evaluate_single_modality_performance((acoustic_features, motion_features, face_features), labels, "多模态特征", subject_ids, original_indices, model_type)
    evaluate_multimodal_voting_performance(acoustic_features, motion_features, face_features, labels, subject_ids, original_indices, model_type)
    print("\n=== 所有评估完成 ===")

# 写一个函数，输入文件名，输出提取出来的音频、Body、Face特征的维度
def print_feature_dimensions(dataDir):
    """打印特征维度"""
    acoustic_feat, _ = extract_audio_features(dataDir)
    motion_feat = extract_body_motion_features(dataDir)
    face_feat = extract_facial_features(dataDir)
    
    print(f"声学特征维度: {np.array(acoustic_feat).shape}")
    print(f"运动特征维度: {np.array(motion_feat).shape}")
    print(f"面部特征维度: {np.array(face_feat).shape}")

if __name__ == "__main__":
    # prefix = '/data/Leo/mm/data/Newborn200/data/'
    # name = '50cm3.16kg1'
    # file = f"{prefix}{name}.wav"
    # print_feature_dimensions(file)

    run_experiment(data_scope="all")
