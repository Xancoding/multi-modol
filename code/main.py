import data_loader
import evaluator
import config
import utils

def display_top_features(feature_names, importance_scores, modality_name, num_features=10):
    """Display the top N features with their importance scores for a given modality."""
    if not feature_names or importance_scores is None:
        return
    
    sorted_features = sorted(zip(feature_names, importance_scores), 
                           key=lambda x: x[1], reverse=True)
    
    print(f"\n{modality_name} Top {num_features} Features:")
    for index, (name, score) in enumerate(sorted_features[:num_features], 1):
        print(f"{index}. {name}: {score:.3f}")

def execute_evaluations(model_evaluator, feature_sets, target_labels, participant_ids, model_type):
    """
    Execute all evaluation methods with clear commenting for enabling/disabling.
    Returns multimodal importance scores for feature analysis.
    """
    # Unpack feature sets for clarity
    audio_features, motion_features, facial_features = feature_sets
    combined_feature_set = (audio_features, motion_features, facial_features)
    
    # ======================================================
    # Evaluation Methods (Enable/disable by commenting)
    # ======================================================
    
    # 1. Audio modality evaluation
    model_evaluator.evaluate_feature_combination(
        audio_features, target_labels, "Audio", participant_ids, model_type
    )

    # 2. Motion modality evaluation
    model_evaluator.evaluate_feature_combination(
        motion_features, target_labels, "Motion", participant_ids, model_type
    )
    
    # 3. Facial modality evaluation
    model_evaluator.evaluate_feature_combination(
        facial_features, target_labels, "Face", participant_ids, model_type
    )
    
    # 4. Feature-level fusion evaluation
    multimodal_importance_scores = model_evaluator.evaluate_feature_combination(
        combined_feature_set, 
        target_labels, 
        "Multimodal-Early Fusion (Concatenation)", 
        participant_ids, 
        model_type
    )

    # # 5. Decision-level fusion evaluation
    # model_evaluator.evaluate_multimodal_fusion(
    #     combined_feature_set, 
    #     target_labels, 
    #     "Multimodal-Late Fusion (Stacking)", 
    #     participant_ids, 
    #     model_type, 
    #     model_type, 
    #     model_type
    # )
    
    # # 6. Conditional fusion evaluation (disabled by default)
    # model_evaluator.evaluate_conditional_fusion(
    #     combined_feature_set, 
    #     target_labels, 
    #     "Multimodal-Conditional Fusion (Entropy Thresholding)",
    #     participant_ids, 
    #     model_type, 
    #     model_type
    # )

    return None

def main():
    """Main function to orchestrate the experiment workflow."""
    enable_feature_printing = False
    utils.initialize_random_seed(config.seed)
    
    # Load dataset
    participant_ids, feature_sets, target_labels, feature_names = data_loader.load_data()
    
    # Initialize model evaluator
    model_evaluator = evaluator.ModelEvaluator()
    
    # Execute evaluations (modify evaluation methods in execute_evaluations as needed)
    multimodal_importance_scores = execute_evaluations(
        model_evaluator, 
        feature_sets[:3],  # audio, motion, facial features
        target_labels,
        participant_ids,
        config.model_type
    )
    
    print("\n=== Evaluation Completed ===")
    
    if config.model_type in ['xgb', 'lgbm', 'rf'] and enable_feature_printing:
        display_top_features(
            sum(feature_names, []),  # Flatten all feature names
            multimodal_importance_scores, 
            "Multimodal", 
            num_features=20
        )

if __name__ == "__main__":
    main()