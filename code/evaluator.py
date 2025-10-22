# Third-party imports
import numpy as np
from tqdm import tqdm

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # 用于指定特征列范围

# Local imports
import config
import random
import torch

class ModelEvaluator:
    """Manages model evaluation workflow with two data splitting strategies"""
    def __init__(self):
        self._set_random_seeds(config.seed)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.cross_validator = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        self.metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1-score']
        # Model configurations
        self.model_configs = {
            'svm': {'class': SVC, 'params': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 
                                              'class_weight': 'balanced', 'probability': True, 'random_state': config.seed}},
            'rf': {'class': RandomForestClassifier, 'params': {'n_estimators': 100, 
                                                            'class_weight': 'balanced', 'random_state': config.seed}},
            'lgbm': {'class': LGBMClassifier, 'params': {'class_weight': 'balanced', 
                                                        'n_estimators': 100, 'verbosity': -1, 'random_state': config.seed}},
        }

    def _set_random_seeds(self, seed):
        """Set all random seeds for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def create_model(self, model_architecture):
        """Instantiate specified classifier type"""
        config = self.model_configs.get(model_architecture.lower())
        return config['class'](**config['params']) if config else None

    def _prepare_features(self, feature_data):
        """Prepare feature data for single or multi-modal input"""
        if isinstance(feature_data, tuple):
            assert all(len(f) == len(feature_data[0]) for f in feature_data)
            return np.concatenate([np.array(f) for f in feature_data], axis=1)
        return np.array(feature_data)

    def _compute_metrics(self, y_true, y_pred):
        """Compute classification metrics"""
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics = {k: report['macro avg'][k] for k in ['precision', 'recall', 'f1-score']}
        metrics.update({'accuracy': report['accuracy']})

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0 

        return metrics

    def _print_metrics(self, metrics):
        """Print evaluation metrics"""
        for k in self.metric_names:
            print(f"{k.capitalize()}: {metrics[k] * 100:.2f}")

    def _train_model(self, X_train, y_train, model, use_sampling=True):
        """Train model with balanced data"""
        scaler = StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)
        if use_sampling:
            ros = RandomOverSampler(random_state=config.seed)
            X_res, y_res = ros.fit_resample(X_scaled, y_train)
            model.fit(X_res, y_res)
        else:
            model.fit(X_scaled, y_train)
        return model, scaler

    def _evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        return self._compute_metrics(y_true, y_pred)

    def evaluate_feature_combination(self, feature_data, target_labels, scenes, method_name, subject_ids, model_architecture='svm', enable_scene_printing=False):
        """Evaluate feature combination (single or multi-modal)"""
        print(f"\n=== Evaluating {method_name} ({model_architecture}) ===")
        X, y, groups = self._prepare_features(feature_data), np.array(target_labels), np.array(subject_ids)
        scenes = np.array(scenes)
        print(f"Original feature dimension: {X.shape[1]}")

        importance = []
        metrics = {k: [] for k in self.metric_names}
        scene_acc_dict = {}  # Store scene accuracy
        
        # Extract subject IDs from group names
        groups = np.array([str(g).split('_')[0] for g in groups])
        
        for train_idx, test_idx in tqdm(self.cross_validator.split(X, y, groups), 
                                    total=config.n_splits, 
                                    desc=f"{method_name} ({model_architecture})"):
            fold_model = self.create_model(model_architecture)
            trained_model, scaler = self._train_model(X[train_idx], y[train_idx], fold_model)

            y_pred = trained_model.predict(scaler.transform(X[test_idx]))  
            # Overall evaluation
            fold_metrics = self._evaluate_model(y[test_idx], y_pred)
            for k in metrics: 
                metrics[k].append(fold_metrics[k])
            
            # Scene evaluation
            if not np.all(scenes == 0):
                for scene in np.unique(scenes[test_idx]):
                    mask = scenes[test_idx] == scene
                    scene_acc = accuracy_score(y[test_idx][mask], y_pred[mask])
                    scene_acc_dict.setdefault(scene, []).append(scene_acc)

            # Feature importance for tree-based models
            if model_architecture.lower() in ['rf', 'lgbm']:
                fold_importance = trained_model.feature_importances_
                fold_importance = np.maximum(fold_importance, 0)
                importance.append(fold_importance / np.sum(fold_importance))  # Normalize
        
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        self._print_metrics(avg_metrics)
        
        # Print scene results
        if scene_acc_dict and enable_scene_printing:
            print("\n=== Scene Accuracy ===")
            for scene in sorted(scene_acc_dict):
                print(f"Scene {scene}: {np.mean(scene_acc_dict[scene]):.4f}")
        
        # Return feature importance if applicable
        if model_architecture.lower() in ['rf', 'lgbm']:
            avg_importance = np.mean(importance, axis=0)
            return avg_importance

        return avg_metrics


    def evaluate_multimodal_fusion(self, feature_data, target_labels, method_name, subject_ids,
                                acoustic_model, motion_model, face_model):
        """
        Evaluate Stacking fusion strategy for multimodal data
        Each base model only uses its corresponding modality features
        """
        print(f"\n=== Evaluating {method_name} ({acoustic_model}, {motion_model}, {face_model}) ===")
        
        # Prepare labels and groups for stratified group cross-validation
        y = np.array(target_labels)
        groups = np.array([str(g).split('_')[0] for g in subject_ids])
        
        try:
            # Unpack feature sets for each modality
            X_acoustic, X_motion, X_face = (np.array(f) for f in feature_data)
            # Get feature dimensions for each modality (used for column indexing)
            n_acoustic = X_acoustic.shape[1]
            n_motion = X_motion.shape[1]
            n_face = X_face.shape[1]
        except ValueError:
            print("Error: feature_data must contain 3 feature sets")
            return None
        
        # Initialize dictionary to store metrics across cross-validation folds
        metrics = {k: [] for k in self.metric_names}
        
        # Iterate through each fold of cross-validation
        for train_idx, test_idx in tqdm(self.cross_validator.split(X_acoustic, y, groups),
                                        total=config.n_splits,
                                        desc=f"Stacking ({method_name})"):
            
            # Split labels into training and test sets for current fold
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 1. Combine modality features for training and test sets
            X_train = np.concatenate([
                X_acoustic[train_idx],  # Acoustic features for training
                X_motion[train_idx],    # Motion features for training
                X_face[train_idx]       # Face features for training
            ], axis=1)
            X_test = np.concatenate([
                X_acoustic[test_idx],   # Acoustic features for testing
                X_motion[test_idx],     # Motion features for testing
                X_face[test_idx]        # Face features for testing
            ], axis=1)
            
            # 2. Define column ranges for each modality in the combined feature matrix
            # Acoustic modality: first n_acoustic columns
            # Motion modality: next n_motion columns
            # Face modality: remaining n_face columns
            acoustic_columns = list(range(n_acoustic))
            motion_columns = list(range(n_acoustic, n_acoustic + n_motion))
            face_columns = list(range(n_acoustic + n_motion, n_acoustic + n_motion + n_face))
            
            # 3. Create dedicated pipelines for each modality
            # Each pipeline includes:
            # - Feature selection (only current modality's columns)
            # - Standardization (only on current modality's features)
            # - Modality-specific classifier
            base_models = [
                ('acoustic', Pipeline([
                    ('select_features', ColumnTransformer([('select', 'passthrough', acoustic_columns)])),
                    ('scaler', StandardScaler()),
                    ('classifier', self.create_model(acoustic_model))
                ])),
                ('motion', Pipeline([
                    ('select_features', ColumnTransformer([('select', 'passthrough', motion_columns)])),
                    ('scaler', StandardScaler()),
                    ('classifier', self.create_model(motion_model))
                ])),
                ('face', Pipeline([
                    ('select_features', ColumnTransformer([('select', 'passthrough', face_columns)])),
                    ('scaler', StandardScaler()),
                    ('classifier', self.create_model(face_model))
                ]))
            ]
            
            # 4. Define meta-classifier (fuses predictions from base models)
            meta_classifier = LogisticRegression(
                class_weight='balanced',
                random_state=config.seed,
                max_iter=1000
            )
            
            # 5. Initialize stacking classifier
            stacking_classifier = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_classifier,
                cv=3,  # Cross-validation for generating base model predictions
                stack_method='predict_proba'  # Use class probabilities as meta-features
            )
            
            # 6. Train stacking classifier and generate predictions
            # Each base model only uses its designated modality features during training
            stacking_classifier.fit(X_train, y_train)
            y_pred = stacking_classifier.predict(X_test)
            
            # 7. Evaluate performance for current fold
            fold_metrics = self._evaluate_model(y_test, y_pred)
            for metric in metrics:
                metrics[metric].append(fold_metrics[metric])
        
        # Calculate average metrics across all cross-validation folds
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        self._print_metrics(avg_metrics)
        return avg_metrics