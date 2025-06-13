from data_loader import load_data
from evaluator import evaluate_modality
import config
import utils

def print_features():
    from features.Feature_Extraction_Audio import acoustic_features_and_spectrogram, NICUWav2Segments
    from features.Feature_Extraction_Body import body_features
    from features.Feature_Extraction_Face import facial_features
    import utils
    dataDir = '/data/Leo/mm/data/Newborn200/data/50cm3.37kg.wav'
    labelDir = utils.get_label_file_path(dataDir)
    sample = NICUWav2Segments(dataDir, labelDir)
    acoustic_feature, audio_feature_names = acoustic_features_and_spectrogram(sample["data"])
    print("Audio Features Shape:", acoustic_feature.shape)
    print("Audio Features Names:", len(audio_feature_names))
    # print("Audio Features Names:", audio_feature_names)
    file = utils.get_motion_feature_file_path(dataDir)
    motion_feature, body_feature_names, _ = body_features(file, config.slidingWindows, config.step, labelDir)
    print("Motion Features Shape:", motion_feature.shape)
    print("Motion Features Names:", len(body_feature_names))
    # print("Body Features Names:", body_feature_names)
    file = utils.get_face_landmark_file_path(dataDir)
    facial_feature, facial_feature_names, _ = facial_features(file, config.slidingWindows, config.step, labelDir)
    print("Facial Features Shape:", facial_feature.shape)
    print("Facial Features Names Shape:", len(facial_feature_names))
    # print("Facial Features Names:", facial_feature_names)

def main():
    """Main experiment function"""
    utils.initialize_random_seed(config.seed)
    
    subject_ids, acoustic_features, motion_features, face_features, labels = load_data()
    
    model_type = 'svm'  # Options: 'svm', 'fnn', 'knn', 'rf', 'xgb', 'lgbm'
    
    evaluate_modality(acoustic_features, labels, "Audio", subject_ids, model_type)
    evaluate_modality(motion_features, labels, "Motion", subject_ids, model_type)
    evaluate_modality(face_features, labels, "Face", subject_ids, model_type)
    evaluate_modality((acoustic_features, motion_features, face_features), labels, "Multimodal", subject_ids, model_type)
    print("\n=== All evaluations completed ===")

if __name__ == "__main__":
    # print_features()
    main()
