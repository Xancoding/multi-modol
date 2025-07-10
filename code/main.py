from data_loader import load_data
from evaluator import ModelEvaluator
import config
import utils
import random

def select_test_cases(subject_ids, hard_cases_num=20, easy_cases_num=20):
    """Select balanced test cases (hard and easy)"""
    random.seed(config.seed)  # Ensure reproducibility
    # Predefined difficult cases
    difficult_cases = {
        "50cm3.16kg", "52cm3.52kg1", "50cm3.2kg", "50cm3.4kg2", "51cm3.25kg",
        "47cm2.9kg", "48cm2.6kg", "48cm2.7kg1", "48cm2.58kg", "48cm3.15kg",
        "48cm3.44kg", "49cm3.4kg", "50cm2.94kg", "50cm2.95kg2", "50cm3.1kg2",
        "50cm3.2kg4", "50cm3.13kg", "50cm3.15kg1", "50cm3.15kg2", "50cm3.22kg",
        "50cm3.22kg1", "50cm3.24kg", "50cm3.44kg", "50cm3.71kg", "50cm3kg3",
        "51cm2.97kg", "51cm3.2kg", "51cm3.55kg", "51cm3.74kg", "51cm4.05kg",
        "52cm3.4kg1", "52cm3.75kg", "53cm3.81kg", "54cm3.9kg"
    }
    all_cases = set(subject_ids)
    easy_cases = list(all_cases - difficult_cases)
    # Random selection with seed
    selected_hard = random.sample(sorted(difficult_cases), min(hard_cases_num, len(difficult_cases)))
    selected_easy = random.sample(sorted(easy_cases), min(easy_cases_num, len(easy_cases)))
    print(f"Total: {len(all_cases)} | Hard: {len(selected_hard)} | Easy: {len(selected_easy)}")
    return selected_easy, selected_hard

def print_top_features(feature_names, importance_scores, modality_name, top_n=10):
    """Print top N important features with their scores"""
    if not feature_names or importance_scores is None:
        return
    
    sorted_features = sorted(zip(feature_names, importance_scores), 
                           key=lambda x: x[1], reverse=True)
    
    print(f"\n{modality_name} Top {top_n} Features:")
    for i, (name, score) in enumerate(sorted_features[:top_n]):
        print(f"{i+1}. {name}: {score:.3f}")

def run_evaluations(evaluator, features, labels, subject_ids, model_type):
    """
    Run all evaluation methods with easy commenting capability.
    Returns multimodal importances for feature importance analysis.
    """
    # Unpack features for clarity
    acoustic, motion, face = features[0], features[1], features[2]
    combined_features = (acoustic, motion, face)
    # combined_features = (motion, face)
    
    # ======================================================
    # Evaluation Methods (Comment/uncomment as needed)
    # ======================================================
    
    # # 1. Audio modality evaluation
    # evaluator.evaluate_feature_combination(
    #     acoustic, labels, "Audio", subject_ids, model_type
    # )

    # # 2. Motion modality evaluation
    # evaluator.evaluate_feature_combination(
    #     motion, labels, "Motion", subject_ids, model_type
    # )
    
    # # 3. Face modality evaluation
    # evaluator.evaluate_feature_combination(
    #     face, labels, "Face", subject_ids, model_type
    # )
    
    # # 5. Decision-level fusion evaluation
    # evaluator.evaluate_multimodal_fusion(
    #     combined_features, 
    #     labels, "Multimodal-Decision Level Fusion (Stacking)", 
    #     subject_ids, model_type, model_type, model_type
    # )

    # # 4. Feature-level fusion evaluation
    # multimodal_importances = evaluator.evaluate_feature_combination(
    #     combined_features, 
    #     labels, "Multimodal-Feature Level Fusion (Concatenation)", subject_ids, model_type
    # )
    
    # # 6. Adaptive fusion evaluation
    evaluator.evaluate_adaptive_multimodal_fusion(
        combined_features, 
        labels, "Adaptive Fusion Strategy", 
        subject_ids, model_type, model_type
    )


    # return multimodal_importances
    return None

def main():
    """Main experiment function"""
    utils.initialize_random_seed(config.seed)
    
    # Load all data
    (subject_ids, features, labels, feature_names) = load_data()

    # Select test cases
    easy_cases, hard_cases = select_test_cases(subject_ids,
                                               config.hard_cases_num, 
                                               config.easy_cases_num)
    
    # Initialize evaluator
    # evaluator = ModelEvaluator()
    evaluator = ModelEvaluator(easy_cases, hard_cases)

    
    # Run evaluations (comment out methods in run_evaluations as needed)
    multimodal_importances = run_evaluations(
        evaluator, 
        features[:3],  # acoustic, motion, face features
        labels,
        subject_ids,
        config.model_type
    )
    
    print("\n=== Evaluation Complete ===")
    
    # # Print feature importance if applicable
    # if config.model_type in ['xgb', 'lgbm', 'rf']:
    #     print_top_features(
    #         sum(feature_names[:3], []),  # Combine all feature names
    #         multimodal_importances, 
    #         "多模态", 
    #         top_n=20
    #     )

if __name__ == "__main__":
    main()