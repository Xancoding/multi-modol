# Third-party imports
import numpy as np
from tqdm import tqdm

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

# PyTorch imports
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# Local imports
import config
from data_loader import FeatureDataset, compute_class_weights

import random
import torch

from sklearn.feature_selection import VarianceThreshold # <-- 新增导入


class ModelEvaluator:
    """统一管理模型评估流程，支持两种数据划分方式"""    
    def __init__(self, easy_subjects=None, hard_subjects=None, ex_test=False):
        self._set_random_seeds(config.seed)
        self.easy_subjects = easy_subjects
        self.hard_subjects = hard_subjects
        self.test_subjects = easy_subjects + hard_subjects if easy_subjects and hard_subjects else None
        if ex_test:
            self.test_subjects = ['01', '02', '03', '04', '05']
        self.cross_validator = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        self.ex_test = ex_test

        self.model_configs = {
            'svm': {'class': SVC, 'params': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 
                                              'class_weight': 'balanced', 'probability': True, 'random_state': config.seed}},
            'rf': {'class': RandomForestClassifier, 'params': {'n_estimators': 100, 
                                                            'class_weight': 'balanced', 'random_state': config.seed}},
            'lgbm': {'class': LGBMClassifier, 'params': {'class_weight': 'balanced', 
                                                        'n_estimators': 100, 'verbosity': -1, 'random_state': config.seed}},
        }

    def _set_random_seeds(self, seed):
        """设置所有随机种子"""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def create_model(self, model_architecture):
        """创建指定类型的分类器"""
        config = self.model_configs.get(model_architecture.lower())
        return config['class'](**config['params']) if config else None

    def _prepare_features(self, feature_data):
        """准备特征数据，支持单模态或多模态拼接"""
        if isinstance(feature_data, tuple):
            return np.concatenate([np.array(f) if not isinstance(f, np.ndarray) else f for f in feature_data], axis=1)
        return np.array(feature_data) if not isinstance(feature_data, np.ndarray) else feature_data

    def create_balanced_data(self, X_train, y_train):
        """创建平衡数据集"""
        training_dataset = FeatureDataset(X_train, y_train)
        class_weights = compute_class_weights(y_train)
        num_classes = len(np.unique(y_train))

        sampler = WeightedRandomSampler(
            class_weights,
            num_samples=num_classes*len(class_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(config.seed)
        )
        
        loader = DataLoader(
            training_dataset,
            batch_size=min(32, len(training_dataset)),
            sampler=sampler,
            num_workers=4
        )
        
        features, labels = [], []
        for batch_f, batch_l in loader:
            features.append(batch_f.numpy())
            labels.append(batch_l.numpy())
        
        return np.concatenate(features), np.concatenate(labels)

    def _compute_metrics(self, y_true, y_pred):
        """计算分类指标"""
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics = {k: report['macro avg'][k] for k in ['precision', 'recall', 'f1-score']}
        metrics.update({'accuracy': report['accuracy']})
        return metrics

    def _split_data_by_subjects(self, groups):
        """根据指定的测试集婴儿划分数据"""
        train_mask = ~np.isin(groups, self.test_subjects)
        if sum(train_mask) == 0:
            raise ValueError("没有足够的训练数据，请检查测试集婴儿划分")
        return train_mask

    def _print_metrics(self, metrics):
        """打印评估指标"""
        for k in ['precision', 'recall', 'accuracy', 'f1-score']:
            print(f"{k.capitalize()}: {metrics[k]:.4f}")

    def _train_model(self, X_train, y_train, model):
        """训练模型"""
        scaler = StandardScaler().fit(X_train)
        X_balanced, y_balanced = self.create_balanced_data(scaler.transform(X_train), y_train)
        model.fit(X_balanced, y_balanced)
        return model, scaler

    def _evaluate_model(self, model, scaler, X_test, y_test):
        """评估模型"""
        y_pred = model.predict(scaler.transform(X_test))
        return self._compute_metrics(y_test, y_pred)

    def evaluate_feature_combination(self, feature_data, target_labels, method_name, subject_ids, model_architecture='svm'):
        """评估特征组合(单模态或多模态直接拼接)"""
        print(f"\n=== Evaluating {method_name} ({model_architecture}) ===")
        model = self.create_model(model_architecture)
        X, y, groups = self._prepare_features(feature_data), np.array(target_labels), np.array(subject_ids)
        print(f"Original feature dimension: {X.shape[1]}")

        if self.test_subjects:
            train_mask = self._split_data_by_subjects(groups)
            model, scaler = self._train_model(X[train_mask], y[train_mask], model)
            
            if self.ex_test:
                # test_mask = np.isin(groups, self.test_subjects)
                # metrics = self._evaluate_model(model, scaler, X[test_mask], y[test_mask])
                # print(f"\n Test Metrics:")
                # self._print_metrics(metrics) 
                for idx, subjects in enumerate(self.test_subjects):
                    test_mask = np.isin(groups, subjects)
                    y_pred = model.predict(scaler.transform(X[test_mask]))
                    correct = (y_pred == y[test_mask]).sum().item()
                    total = test_mask.sum().item()
                    print(f"{subjects}:{correct}/{total}")
                    print("Predict:", y_pred)  # 打印模型预测结果
                    print("Label:", y[test_mask])  # 打印真实标签
            else:
                for test_name, test_subjects in {'easy': self.easy_subjects, 'hard': self.hard_subjects, 'all': self.test_subjects}.items():
                    test_mask = np.isin(groups, test_subjects)
                    
                    metrics = self._evaluate_model(model, scaler, X[test_mask], y[test_mask])
                    print(f"\n{test_name.capitalize()} Test Metrics:")
                    self._print_metrics(metrics)                

            if model_architecture.lower() in ['rf', 'lgbm']:
                return model.feature_importances_ / np.sum(model.feature_importances_)

            return None
        else:
            importance = []
            metrics = {k: [] for k in ['precision', 'recall', 'f1-score', 'accuracy']}
            for train_idx, test_idx in tqdm(self.cross_validator.split(X, y, groups), total=config.n_splits, desc=f"{method_name} ({model_architecture})"):
                model, scaler = self._train_model(X[train_idx], y[train_idx], model)
                fold_metrics = self._evaluate_model(model, scaler, X[test_idx], y[test_idx])
                for k in metrics: metrics[k].append(fold_metrics[k])

                if model_architecture.lower() in ['rf', 'lgbm']:
                    importance.append(model.feature_importances_ / np.sum(model.feature_importances_))
            
            avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
            self._print_metrics(avg_metrics)
            
            if model_architecture.lower() in ['rf', 'lgbm']:
                avg_importance = np.mean(importance, axis=0)
                return avg_importance

            return None

    def evaluate_multimodal_fusion(self, feature_data, target_labels, method_name, subject_ids, acoustic_model, motion_model, face_model):
        """评估多模态融合(stacking方法)"""
        print(f"\n=== Evaluating {method_name} ({acoustic_model}, {motion_model}, {face_model}) ===")
        X_audio = self._prepare_features(feature_data[0])
        X_motion = self._prepare_features(feature_data[1])
        X_face = self._prepare_features(feature_data[2])
        y = np.array(target_labels)
        groups = np.array(subject_ids)

        def train_fold(train_idx):
            scalers = {}
            X_train = {'audio': X_audio[train_idx], 'motion': X_motion[train_idx], 'face': X_face[train_idx]}
            y_train = y[train_idx]
            
            base_models = []
            for modality, model_arch in zip(['audio', 'motion', 'face'], [acoustic_model, motion_model, face_model]):
                model = self.create_model(model_arch)
                if model_arch in ['svm', 'knn']:
                    model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
                
                # Standardize and train using _train_model
                model, scaler = self._train_model(X_train[modality], y_train, model)
                scalers[modality] = scaler
                base_models.append((modality, model))
            
            # Prepare stacked features
            X_stacked = np.hstack([scalers[m].transform(X_train[m]) for m in ['audio', 'motion', 'face']])
            
            # Train stacking model using _train_model
            stacking_model = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(max_iter=1000, random_state=config.seed),
                cv=3,
                stack_method='predict_proba',
                n_jobs=-1
            )
            stacking_model, _ = self._train_model(X_stacked, y_train, stacking_model)
            
            return {'base_models': base_models, 'stacking_model': stacking_model, 'scalers': scalers}

        def evaluate_fold(test_idx, trained_models):
            X_test = {m: trained_models['scalers'][m].transform(d[test_idx]) for m, d in zip(['audio', 'motion', 'face'], [X_audio, X_motion, X_face])}
            y_pred = trained_models['stacking_model'].predict(np.hstack([X_test[m] for m in ['audio', 'motion', 'face']]))
            return self._compute_metrics(y[test_idx], y_pred)

        if self.test_subjects:
            train_mask = self._split_data_by_subjects(groups)
            trained_models = train_fold(train_mask)
            
            for test_name, test_subjects in {'easy': self.easy_subjects, 'hard': self.hard_subjects, 'all': self.test_subjects}.items():
                test_mask = np.isin(groups, test_subjects)
                if sum(test_mask) == 0:
                    print(f"警告：{test_name}测试集为空")
                    continue
                
                metrics = evaluate_fold(test_mask, trained_models)
                print(f"\n{test_name.capitalize()} Test Metrics:")
                self._print_metrics(metrics)
            return metrics
        else:
            metrics = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1-score']}
            for train_idx, test_idx in tqdm(self.cross_validator.split(X_audio, y, groups), total=config.n_splits, desc=f"{method_name}"):
                trained_models = train_fold(train_idx)
                fold_metrics = evaluate_fold(test_idx, trained_models)
                for k in metrics: metrics[k].append(fold_metrics[k])
            
            avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
            self._print_metrics(avg_metrics)
            return avg_metrics

    def _calculate_entropy(self, probabilities):
        """计算给定概率分布的熵。"""
        # 避免log(0)
        probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
        return -np.sum(probabilities * np.log2(probabilities), axis=1)

    def evaluate_conditional_fusion(self, feature_data, target_labels, method_name, subject_ids, 
                                    model_A_arch, model_B_arch):
        print(f"\n=== Evaluating Conditional Fusion ({method_name}) ===")
        print(f"  Model A (Audio+Motion+Face): {model_A_arch}, Model B (Motion+Face): {model_B_arch}")

        X_audio = self._prepare_features(feature_data[0])
        X_motion = self._prepare_features(feature_data[1])
        X_face = self._prepare_features(feature_data[2])
        
        # New: Prepare X for Model A (audio + motion + face)
        X_all_modalities = self._prepare_features((X_audio, X_motion, X_face)) 
        
        # X for Model B (motion + face) remains the same
        X_motion_face = self._prepare_features((X_motion, X_face)) 
        
        y = np.array(target_labels)
        groups = np.array(subject_ids)

        def train_conditional_models(train_idx):
            # Train Model A (All Modalities: Audio + Motion + Face)
            model_A = self.create_model(model_A_arch)
            if not hasattr(model_A, 'predict_proba'):
                model_A = CalibratedClassifierCV(model_A, cv=3, method='isotonic' if model_A_arch == 'svm' else 'sigmoid')
            
            model_A, scaler_A = self._train_model(X_all_modalities[train_idx], y[train_idx], model_A)

            # Train Model B (Motion + Face)
            model_B = self.create_model(model_B_arch)
            if not hasattr(model_B, 'predict_proba'):
                model_B = CalibratedClassifierCV(model_B, cv=3, method='isotonic' if model_B_arch == 'svm' else 'sigmoid')
            
            model_B, scaler_B = self._train_model(X_motion_face[train_idx], y[train_idx], model_B)
            
            return {
                'model_A': model_A, 'scaler_A': scaler_A,
                'model_B': model_B, 'scaler_B': scaler_B
            }

        def evaluate_conditional_models(test_idx, trained_models):
            model_A = trained_models['model_A']
            scaler_A = trained_models['scaler_A']
            model_B = trained_models['model_B']
            scaler_B = trained_models['scaler_B']

            # Prepare test data for Model A (All Modalities)
            X_all_modalities_test = scaler_A.transform(X_all_modalities[test_idx])
            # Prepare test data for Model B (Motion + Face)
            X_motion_face_test = scaler_B.transform(X_motion_face[test_idx])
            
            y_true_fold = y[test_idx]

            # Get probabilities from Model A (All Modalities)
            proba_A = model_A.predict_proba(X_all_modalities_test)
            entropy_A = self._calculate_entropy(proba_A) # Entropy for Model A

            # Get probabilities from Model B (Motion + Face)
            proba_B = model_B.predict_proba(X_motion_face_test)
            entropy_B = self._calculate_entropy(proba_B) # Entropy for Model B

            # Get predictions from Model A and Model B
            pred_A = model_A.predict(X_all_modalities_test)
            pred_B = model_B.predict(X_motion_face_test)

            # Initialize final predictions
            y_pred_final = np.empty_like(pred_A) # Create an empty array to fill

            # Apply new conditional logic: choose the prediction with lower entropy
            # This creates a boolean mask where True means Model B's entropy is lower
            choose_model_B_mask = entropy_B < entropy_A

            # Where Model B's entropy is lower, use Model B's prediction
            y_pred_final[choose_model_B_mask] = pred_B[choose_model_B_mask]
            # Otherwise (where Model A's entropy is lower or equal), use Model A's prediction
            y_pred_final[~choose_model_B_mask] = pred_A[~choose_model_B_mask]
            
            return self._compute_metrics(y_true_fold, y_pred_final)
        

        if self.test_subjects:
            train_mask = self._split_data_by_subjects(groups)
            trained_models = train_conditional_models(train_mask)
            
            if self.ex_test:
                test_mask = np.isin(groups, self.test_subjects)
                metrics = evaluate_conditional_models(test_mask, trained_models)
                print(f"\n Test Metrics:")
                self._print_metrics(metrics) 
                # for idx, subjects in enumerate(self.test_subjects):
                #     test_mask = np.isin(groups, subjects)
                #     y_pred = model.predict(scaler.transform(X[test_mask]))
                #     print(subjects)
                #     print("Predict:", y_pred)  # 打印模型预测结果
                #     print("Label:", y[test_mask])  # 打印真实标签
            else:            
                for test_name, test_subjects in {'easy': self.easy_subjects, 'hard': self.hard_subjects, 'all': self.test_subjects}.items():
                    test_mask = np.isin(groups, test_subjects)
                    if sum(test_mask) == 0:
                        print(f"警告：{test_name}测试集为空")
                        continue
                    
                    metrics = evaluate_conditional_models(test_mask, trained_models)
                    print(f"\n{test_name.capitalize()} Test Metrics:")
                    self._print_metrics(metrics)
                return metrics
        else:
            metrics = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1-score']}
            for train_idx, test_idx in tqdm(self.cross_validator.split(X_audio, y, groups), total=config.n_splits, desc=f"{method_name}"):
                trained_models = train_conditional_models(train_idx)
                fold_metrics = evaluate_conditional_models(test_idx, trained_models)
                for k in metrics: metrics[k].append(fold_metrics[k])
            
            avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
            self._print_metrics(avg_metrics)
            return avg_metrics