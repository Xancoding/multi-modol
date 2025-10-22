# Standard library imports
import os
import glob

# Third-party imports
import numpy as np
from tqdm import tqdm

# Local imports
from features.Feature_Extraction_Audio import acoustic_features_and_spectrogram, NICUWav2Segments
from features.Feature_Extraction_Body import body_features
from features.Feature_Extraction_Face import facial_features, _face_stats
import config
import utils

from excluded_files import EXCLUDED_NEWBORN200_FILES, EXCLUDED_NICU50_FILES

def extract_audio_features(dataDir):
    """Extract audio features"""
    labelDir = utils.get_label_file_path(dataDir)
    sceneDir = utils.get_scene_file_path(dataDir)
    if not os.path.exists(sceneDir):
        sample = NICUWav2Segments(dataDir, labelDir, None)
        sample['scene'] = [0] * len(sample['label']) 
    else:
        sample = NICUWav2Segments(dataDir, labelDir, sceneDir)
    acoustic_feature, acoustic_feature_name = acoustic_features_and_spectrogram(sample["data"])
    return acoustic_feature, acoustic_feature_name, sample["label"], sample["scene"]

def extract_body_motion_features(dataDir):
    """Extract body motion features"""
    file = utils.get_motion_feature_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    motion_feature, motion_feature_name = body_features(file, config.slidingWindows, config.step, labelDir)
    return motion_feature, motion_feature_name

def extract_facial_features(dataDir):
    """Extract facial features"""
    file = utils.get_face_landmark_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    facial_feature, facial_feature_name = facial_features(file, config.slidingWindows, config.step, labelDir)
    return facial_feature, facial_feature_name

def load_or_extract_features(audio_files):
    """
    Loads or extracts feature data from audio files, including feature names.
    
    Args:
        audio_files (list): List of paths to audio files (WAV format)
    
    Returns:
        dict: Dictionary mapping subject IDs to lists of feature dictionaries
    """
    # Get parent directory and create features directory
    parent_dir = os.path.dirname(audio_files[0])
    features_dir = os.path.join(os.path.dirname(parent_dir), 'Features')
    os.makedirs(features_dir, exist_ok=True)
    
    # Initialize dictionary to store subject features
    subject_features = {}
    
    # Process each audio file with progress bar
    for file_path in tqdm(audio_files, desc="Processing audio files"):
        # Generate feature file path
        feature_file_path = os.path.join(features_dir, utils.generate_feature_filename(file_path))
        
        # Try to load existing features
        if os.path.exists(feature_file_path):
            try:
                # Load saved feature data
                loaded_data = np.load(feature_file_path, allow_pickle=True)
                subject_id = utils.extract_subject_id(file_path)
                
                # Store loaded features in subject dictionary
                subject_features.setdefault(subject_id, []).append({
                    'acoustic': loaded_data['acoustic'],
                    'motion': loaded_data['motion'],
                    'face': loaded_data['face'],
                    'acoustic_feature_names': loaded_data['acoustic_feature_names'].tolist(),
                    'motion_feature_names': loaded_data['motion_feature_names'].tolist(),
                    'face_feature_names': loaded_data['face_feature_names'].tolist(),
                    'label': loaded_data['label'],  
                    'scene': loaded_data['scene']             
                })
                continue
            except Exception as error:
                print(f"Failed to load features, regenerating: {error}")

        # Extract new features
        acoustic_features, acoustic_feature_names, label, scene = extract_audio_features(file_path)
        motion_features, motion_feature_names = extract_body_motion_features(file_path)
        facial_features, facial_feature_names = extract_facial_features(file_path)       

        # Save extracted features
        np.savez(
            feature_file_path,
            acoustic=acoustic_features,
            motion=motion_features,
            face=facial_features,
            acoustic_feature_names=acoustic_feature_names,
            motion_feature_names=motion_feature_names,
            face_feature_names=facial_feature_names,
            label=label,
            scene=scene
        )

        # Store extracted features in subject dictionary
        subject_id = utils.extract_subject_id(file_path)
        subject_features.setdefault(subject_id, []).append({
            'acoustic': acoustic_features,
            'motion': motion_features,
            'face': facial_features,
            'acoustic_feature_names': acoustic_feature_names,
            'motion_feature_names': motion_feature_names,
            'face_feature_names': facial_feature_names,
            'label': label,
            'scene': scene
        })
    
    if _face_stats['total'] > 0:
        print(f"Face Acc: {1 - _face_stats['errors']/_face_stats['total']:.1%}")

    return subject_features

def prepare_feature_matrices(subject_data, enable_scene_printing=False):
    """Prepare feature matrices and print dataset statistics."""
    print("\nPreparing feature matrices...")
    
    # Lists to store features and labels
    acoustic_features = []
    motion_features = []
    facial_features = []
    labels = []
    scenes = []
    subject_ids = []
    
    # Get feature names (assumes all samples have the same feature names)
    first_sample = next(iter(subject_data.values()))[0]
    acoustic_feature_names = first_sample.get('acoustic_feature_names', None)
    motion_feature_names = first_sample.get('motion_feature_names', None)
    facial_feature_names = first_sample.get('face_feature_names', None)
    
    if None in [acoustic_feature_names, motion_feature_names, facial_feature_names]:
        raise ValueError("Feature names not properly loaded! Please check the feature extraction function.")
    
    # Track infant crying status by base ID
    crying_infants = set()
    non_crying_infants = set()
    infant_crying_status = {}  # Maps base infant ID to whether they have cried
    
    # Track segment crying statistics
    crying_segment_count = 0
    non_crying_segment_count = 0
    
    # Track scene distribution
    scene_counts = {}  # {scene_type: count}
    
    for subj_id_with_timestamp, samples in tqdm(subject_data.items(), desc="Processing subjects"):
        # Extract base infant ID (remove timestamp)
        # if np.all(samples[0]['scene'] == 0):
        #     print(subj_id_with_timestamp)
        base_infant_id = subj_id_with_timestamp.split('_')[0]
        
        # Initialize infant crying status if not already recorded
        if base_infant_id not in infant_crying_status:
            infant_crying_status[base_infant_id] = False
        
        has_crying = False
        for sample in samples:
            frame_count = len(sample['label'])
            for i in range(frame_count):
                label = sample['label'][i]
                scene = sample['scene'][i]  
                
                if scene not in scene_counts:
                    scene_counts[scene] = 0
                scene_counts[scene] += 1
                
                if label == 1:  # 1 indicates crying
                    crying_segment_count += 1
                    has_crying = True
                else:
                    non_crying_segment_count += 1
                                
                # Append features and labels for each modality
                acoustic_features.append(sample['acoustic'][i])
                motion_features.append(sample['motion'][i])
                facial_features.append(sample['face'][i])
                labels.append(label)
                scenes.append(scene)
                subject_ids.append(subj_id_with_timestamp)
        
        # Update infant crying status
        if has_crying:
            infant_crying_status[base_infant_id] = True
    
    # Count infants based on crying status
    for base_infant_id, has_cried in infant_crying_status.items():
        if has_cried:
            crying_infants.add(base_infant_id)
        else:
            non_crying_infants.add(base_infant_id)
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total infants: {len(infant_crying_status)}")
    print(f"Crying infants: {len(crying_infants)}")
    print(f"Non-crying infants: {len(non_crying_infants)}")
    print(f"\nTotal segments: {len(labels)}")
    print(f"Crying segments: {crying_segment_count}")
    print(f"Non-crying segments: {non_crying_segment_count}")
    print(f"Crying segment proportion: {crying_segment_count/len(labels):.2%}")

    print("\n=== Feature Dimensions ===")
    print(f"Acoustic features: {len(acoustic_feature_names)}")
    print(f"Motion features: {len(motion_feature_names)}")
    print(f"Facial features: {len(facial_feature_names)}")
    
    if not np.all(np.array(scenes) == 0) and enable_scene_printing:
        print("\n=== Scene Distribution ===")
        total_scenes = sum(scene_counts.values())
        for scene in sorted(scene_counts.keys()):
            count = scene_counts[scene]
            print(f"Scene {scene}: {count} segments ({count/total_scenes:.2%})")
    
    return (
        subject_ids,
        acoustic_features,
        motion_features,
        facial_features,
        acoustic_feature_names,
        motion_feature_names,
        facial_feature_names,
        labels,
        scenes,
    )

def load_data(enable_scene_printing=False):
    """Load and prepare all data"""
    prefix = config.dataDir
    ori_wav_files = glob.glob(os.path.join(prefix, "*.wav"))
    
    if 'NEWBORN200' in config.dataDir:
        excluded_wav_files = []        
        # excluded_wav_files = [prefix + x + '.wav' for x in EXCLUDED_NEWBORN200_FILES]
    elif 'NICU50' in config.dataDir:
        excluded_wav_files = [prefix + x + '.wav' for x in EXCLUDED_NICU50_FILES]
    else:
        excluded_wav_files = []
    wav_files = [f for f in ori_wav_files if f not in excluded_wav_files]

    # wav_files = [
    #     prefix + x + '.wav' for x in
    #     [
    #         'lxm-baby-m_2025-07-29-14-19-07',
    #         # 'zm-baby-m_2025-07-23-19-24-10',
    #         'lxm-baby-m_2025-07-29-14-55-30',
    #     ]
    # ]

    subject_data = load_or_extract_features(wav_files)
    print(f"Total subjects: {len(subject_data)}")
    
    (subject_ids, acoustic_features, motion_features, 
     face_features, acoustic_feature_names,
     motion_feature_names, face_feature_names, labels, scenes) = prepare_feature_matrices(subject_data, enable_scene_printing)

    # # only EAR
    # face_feature_names = face_feature_names[5:]
    # face_features[:] = [row[5:] for row in face_features]   

    # # only MAR
    # face_feature_names = face_feature_names[:5]
    # face_features[:] = [row[:5] for row in face_features]

    return (
        subject_ids,
        (acoustic_features, motion_features, face_features),  
        (acoustic_feature_names, motion_feature_names, face_feature_names),
        labels,
        scenes,
    )