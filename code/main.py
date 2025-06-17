from data_loader import load_data
from evaluator import evaluate_modality
import config
import utils
import numpy as np

def print_top_features(feature_names, importance_scores, modality_name, top_n=10):
    """打印前N个重要特征"""
    if len(feature_names) == 0 or importance_scores is None:
        return
        
    # 合并特征名和重要性分数
    feature_importance = list(zip(feature_names, importance_scores))
    # 按重要性排序(降序)
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{modality_name}模态前{top_n}重要特征:")
    for i, (name, score) in enumerate(feature_importance[:top_n]):
        print(f"{i+1}. {name}: {score:.4f}")

def print_features():
    """打印各模态特征维度(调试用)"""
    from features.Feature_Extraction_Audio import acoustic_features_and_spectrogram, NICUWav2Segments
    from features.Feature_Extraction_Body import body_features
    from features.Feature_Extraction_Face import facial_features
    import utils
    
    dataDir = '/data/Leo/mm/data/Newborn200/data/50cm3.37kg.wav'
    labelDir = utils.get_label_file_path(dataDir)
    sample = NICUWav2Segments(dataDir, labelDir)
    
    acoustic_feature, audio_feature_names = acoustic_features_and_spectrogram(sample["data"])
    print("音频特征维度:", acoustic_feature.shape)
    print("音频特征数量:", len(audio_feature_names))
    
    file = utils.get_motion_feature_file_path(dataDir)
    motion_feature, body_feature_names, _ = body_features(file, config.slidingWindows, config.step, labelDir)
    print("运动特征维度:", motion_feature.shape)
    print("运动特征数量:", len(body_feature_names))
    
    file = utils.get_face_landmark_file_path(dataDir)
    facial_feature, facial_feature_names, _ = facial_features(file, config.slidingWindows, config.step, labelDir)
    print("面部特征维度:", facial_feature.shape)
    print("面部特征数量:", len(facial_feature_names))

def main():
    """主实验函数"""
    utils.initialize_random_seed(config.seed)
    
    # 加载数据
    (subject_ids, acoustic_features, motion_features, 
     face_features, labels, acoustic_feature_names,
     motion_feature_names, face_feature_names) = load_data()
    
    model_type = 'lgbm'  # 可选: 'svm', 'fnn', 'knn', 'rf', 'xgb', 'lgbm'
    
    # 评估单模态
    print("\n" + "="*50)
    print("开始单模态评估")
    print("="*50)
    
    audio_metrics, audio_importances = evaluate_modality(
        acoustic_features, labels, "音频", subject_ids, model_type)
    
    motion_metrics, motion_importances = evaluate_modality(
        motion_features, labels, "运动", subject_ids, model_type)
    
    face_metrics, face_importances = evaluate_modality(
        face_features, labels, "面部", subject_ids, model_type)
        
    # 评估多模态
    print("\n" + "="*50)
    print("开始多模态评估")
    print("="*50)
    
    multimodal_metrics, multimodal_importances = evaluate_modality(
        (acoustic_features, motion_features, face_features), 
        labels, "多模态", subject_ids, model_type)
    
    # 合并所有特征名称
    all_feature_names = (acoustic_feature_names + 
                        motion_feature_names + 
                        face_feature_names)
    
    print("\n=== 所有评估完成 ===")

    # if model_type in ['xgb', 'lgbm', 'rf']:
    #     print_top_features(acoustic_feature_names, audio_importances, "音频")
    #     print_top_features(motion_feature_names, motion_importances, "运动")
    #     print_top_features(face_feature_names, face_importances, "面部")
    #     print_top_features(all_feature_names, multimodal_importances, "多模态")
    # else:
    #     print("没有可用的特征重要性分数，无法打印重要特征。")    

if __name__ == "__main__":
    # print_features()  # 调试时取消注释
    main()