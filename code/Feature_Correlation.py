import numpy as np
import matplotlib.pyplot as plt
import os
from features.Feature_Extraction_Audio import extract_raw_acoustic_features
from features.Feature_Extraction_Body import extract_raw_motion_features
from features.Feature_Extraction_Face import extract_raw_face_features
from scipy.stats import spearmanr # 导入 spearmanr

def create_cross_modality_correlation(feat1, names1, feat2, names2, name1, name2):
    num_samples = feat1.shape[0]
    num_features1 = feat1.shape[1]
    num_features2 = feat2.shape[1]
    feature_names = [f"corr_{name1}_{n1}_vs_{name2}_{n2}" for n1 in names1 for n2 in names2]
    correlation_features = np.zeros((num_samples, len(feature_names)))
    for sample_idx in range(num_samples):
        sample_feat1 = feat1[sample_idx].T
        sample_feat2 = feat2[sample_idx].T
        correlations = []
        for i in range(num_features1):
            for j in range(num_features2):
                ts1 = sample_feat1[:, i]
                ts2 = sample_feat2[:, j]
                # 检查序列长度和标准差，避免计算错误
                if len(ts1) > 1 and len(ts2) > 1 and np.std(ts1) > 0 and np.std(ts2) > 0:
                    # 使用斯皮尔曼相关系数
                    corr, _ = spearmanr(ts1, ts2)
                else:
                    corr = 0.0
                correlations.append(corr)
        correlation_features[sample_idx] = correlations
    return correlation_features, feature_names

def down_sample(sample, target):
    n, d, origin = sample.shape
    out = np.zeros((n, d, target))
    idx = np.linspace(0, origin, target + 1, dtype=int)
    for i in range(target):
        start = idx[i]
        end = idx[i+1]
        if start < end:
            out[:, :, i] = np.mean(sample[:, :, start:end], axis=2)
        else:
            if i > 0:
                out[:, :, i] = out[:, :, i-1]
            else:
                out[:, :, i] = np.zeros((n, d))
    return out

def plot_time_series_features(idx, features, feature_names, title="Time Series Features", selected_feature_names=None):
    data_to_plot = features[idx, :, :] 

    if selected_feature_names:
        name_to_index = {name: i for i, name in enumerate(feature_names)}
        indices_to_plot = []
        features_to_display_names = []
        for name in selected_feature_names:
            if name in name_to_index:
                indices_to_plot.append(name_to_index[name])
                features_to_display_names.append(name)

        if not indices_to_plot:
            return

        data_to_plot = data_to_plot[indices_to_plot, :] 
    else:
        indices_to_plot = list(range(len(feature_names)))
        features_to_display_names = feature_names

    num_features_to_display = len(indices_to_plot)
    time_steps = data_to_plot.shape[1]
    time_axis = np.arange(time_steps)

    if num_features_to_display == 0:
        return

    fig, axes = plt.subplots(nrows=num_features_to_display, ncols=1, 
                             figsize=(15, max(3, 2.5 * num_features_to_display)), sharex=True)
    
    if num_features_to_display == 1:
        axes = [axes]

    for i in range(num_features_to_display): 
        ax = axes[i]
        ax.plot(time_axis, data_to_plot[i, :]) 
        ax.set_ylabel(features_to_display_names[i], fontsize=20) 
        ax.grid(True)
        ax.tick_params(axis='y', labelsize=8) 

    axes[-1].set_xlabel("Time Steps", fontsize=20) 
    fig.suptitle(title, fontsize=40) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 

    output_dir = 'img'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, title.replace(" ", "_") + '.png'))
    plt.show()

def main():
    prefix = "/data/Leo/mm/data/Newborn200/data/" 
    baby = "04.wav" 
    file_path = prefix + baby

    motion_feat, motion_names = extract_raw_motion_features(file_path)
    face_feat, face_names = extract_raw_face_features(file_path)
    audio_feat, label, audio_names = extract_raw_acoustic_features(file_path)

    target_time_steps = min(motion_feat.shape[2], face_feat.shape[2], audio_feat.shape[2])
    audio_feat = down_sample(audio_feat, target_time_steps)
    motion_feat = down_sample(motion_feat, target_time_steps) 
    face_feat = down_sample(face_feat, target_time_steps) 
    
    # audio_motion_corr, audio_motion_names = create_cross_modality_correlation(
    #     audio_feat, audio_names, 
    #     motion_feat, motion_names,
    #     "audio", "motion"
    # )
    # audio_face_corr, audio_face_names = create_cross_modality_correlation(
    #     audio_feat, audio_names,
    #     face_feat, face_names,
    #     "audio", "face"
    # )
    # motion_face_corr, motion_face_names = create_cross_modality_correlation(
    #     motion_feat, motion_names,
    #     face_feat, face_names,
    #     "motion", "face"
    # )
    # all_cross_features = np.concatenate([
    #     audio_motion_corr,
    #     # audio_face_corr,
    #     # motion_face_corr
    # ], axis=1)
    # all_cross_names = audio_motion_names 
    # # + audio_face_names + motion_face_names
    # print(all_cross_features.shape, len(all_cross_names))



    selected_audio = ['chroma_5', 'spectral_spread']
    selected_motion = ["Left-arm", "Right-arm"]
    selected_face = ['mars', 'left_ears', 'right_ears']
    multi_feat = np.concatenate((audio_feat, motion_feat, face_feat), axis=1)
    multi_name = audio_names + motion_names + face_names
    multi_selected_plot_names = selected_audio + selected_motion + selected_face
    idx = 0
    plot_time_series_features(idx, multi_feat, multi_name, 
                              title="multimodal", 
                              selected_feature_names=multi_selected_plot_names) 
    # plot_time_series_features(idx, audio_feat, audio_names, 
    #                         title="audio",
    #                         # selected_feature_names=selected_audio
    #                         )
    # plot_time_series_features(idx, motion_feat, motion_names,
    #                         title="motion",
    #                         # selected_feature_names=selected_motion
    #                         )
    # plot_time_series_features(idx, face_feat, face_names,
    #                         title="face",
    #                         # selected_feature_names=selected_face
    #                         )
    
if __name__ == "__main__":
    main()