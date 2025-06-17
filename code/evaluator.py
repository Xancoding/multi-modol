# Third-party imports
import numpy as np

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# PyTorch imports
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# Local imports
import config
from data_loader import FeatureDataset, compute_class_weights

def create_model(model_architecture, random_state=None):
    """Create specified classifier"""
    if model_architecture == 'svm':
        return SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=random_state, probability=True)
    elif model_architecture == 'fnn':
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    elif model_architecture == 'knn':
        return KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    elif model_architecture == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    elif model_architecture == 'xgb':
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state, scale_pos_weight=1)
    elif model_architecture == 'lgbm':
        return LGBMClassifier(random_state=random_state, class_weight='balanced', n_estimators=100, verbosity=-1)
    raise ValueError(f"Unknown model architecture: {model_architecture}")

def compute_performance_metrics(true_labels, predicted_labels):
    """Calculate classification metrics"""
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score']
    }

def cross_validate_model(feature_data, target_labels, subject_ids, model_name, model_architecture='svm'):
    """使用加权随机采样评估分类器性能"""
    performance_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    
    # 统一处理输入特征
    if isinstance(feature_data, tuple):
        # 处理多模态情况(元组)
        feature_list = [np.array(f) if not isinstance(f, np.ndarray) else f for f in feature_data]
        feature_matrix = np.concatenate(feature_list, axis=1)
        total_features = feature_matrix.shape[1]
    else:
        # 处理单模态情况
        feature_matrix = np.array(feature_data) if not isinstance(feature_data, np.ndarray) else feature_data
        total_features = feature_matrix.shape[1]
    
    label_array = np.array(target_labels)
    subject_groups = np.array(subject_ids)
    
    cross_validator = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    
    # 初始化特征重要性
    importance_scores = None
    
    for fold, (train_indices, test_indices) in enumerate(cross_validator.split(feature_matrix, label_array, subject_groups)):        
        X_train, X_test = feature_matrix[train_indices], feature_matrix[test_indices]
        y_train, y_test = label_array[train_indices], label_array[test_indices]
        
        # 标准化特征
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 创建加权采样器
        training_dataset = FeatureDataset(X_train_scaled, y_train)
        class_weights = compute_class_weights(y_train)
        num_classes = len(np.unique(y_train))
        balanced_sampler = WeightedRandomSampler(class_weights, num_samples=num_classes*len(class_weights), replacement=True)

        # 创建DataLoader
        training_loader = DataLoader(
            training_dataset,
            batch_size=min(32, len(training_dataset)),
            sampler=balanced_sampler,
            num_workers=4
        )
        
        # 转换为numpy数组
        balanced_features = []
        balanced_labels = []
        for batch_features, batch_labels in training_loader:
            balanced_features.append(batch_features.numpy())
            balanced_labels.append(batch_labels.numpy())
        
        X_train_balanced = np.concatenate(balanced_features)
        y_train_balanced = np.concatenate(balanced_labels)
        
        # 训练模型
        model = create_model(model_architecture, config.seed)
        model.fit(X_train_balanced, y_train_balanced)
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            fold_importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            fold_importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            fold_importances = np.zeros(total_features)
        
        if importance_scores is None:
            importance_scores = fold_importances
        else:
            importance_scores += fold_importances
        
        # 评估
        y_pred = model.predict(X_test_scaled)
        performance_results.update(compute_performance_metrics(y_test, y_pred))
    
    # 标准化重要性分数(确保使用浮点除法)
    if importance_scores is not None:
        importance_scores = importance_scores / float(config.n_splits)  # 显式转换为浮点除法
    
    average_metrics = {k: np.mean(v) for k, v in performance_results.items()}
    print(f"\n{model_name} ({model_architecture.upper()}) results:")
    print(f"Accuracy: {average_metrics['accuracy']:.4f}")
    print(f"Precision: {average_metrics['precision']:.4f}")
    print(f"Recall: {average_metrics['recall']:.4f}")
    print(f"F1-score: {average_metrics['f1-score']:.4f}")    
    
    return average_metrics, importance_scores

def evaluate_modality(feature_data, target_labels, modality_name, subject_ids, model_architecture='svm'):
    print(f"\n=== Evaluating {modality_name} features ({model_architecture.upper()}) ===")
    return cross_validate_model(feature_data, target_labels, subject_ids, f"{modality_name} model", model_architecture)