import json
import numpy as np
import glob 

def calculate_mar(landmarks):
    """计算嘴部纵横比"""
    p48, p51, p54, p57, p62, p66 = [landmarks[i] for i in [48, 51, 54, 57, 62, 66]]
    mouth_width = np.linalg.norm(p54 - p48)
    upper_lip = np.linalg.norm(p62 - p51)
    lower_lip = np.linalg.norm(p66 - p57)
    return (upper_lip + lower_lip) / (2 * mouth_width)

def calculate_ear(landmarks, eye_points):
    """计算眼睛闭合度"""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
    return (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2 * np.linalg.norm(p1-p4))

def process_json(json_path, output_path=None):
    """
    处理整个JSON文件
    参数：
        json_path: 输入的JSON文件路径
        output_path: 输出结果路径（可选）
    返回：
        包含所有帧特征的列表
    """
    with open(json_path) as f:
        data = json.load(f)
    
    # 转换为numpy数组加速计算
    frames = []
    for frame in data['frames']:
        if not frame['landmarks']:
            frames.append(None)
            continue
        frames.append(np.array(frame['landmarks']))
    
    # 计算特征
    results = []
    for i, landmarks in enumerate(frames):
        if landmarks is None:
            results.append({'frame': i, 'mar': None, 'ear_left': None, 'ear_right': None})
            continue
        
        try:
            results.append({
                'frame': i,
                'mar': float(calculate_mar(landmarks)),
                'ear_left': float(calculate_ear(landmarks, [36, 37, 38, 39, 40, 41])),
                'ear_right': float(calculate_ear(landmarks, [42, 43, 44, 45, 46, 47]))
            })
        except Exception:
            results.append({'frame': i, 'mar': None, 'ear_left': None, 'ear_right': None})
    
    # 保存结果
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'video_info': data['video_info'],
                'features': results
            }, f, indent=2)
    
    return results

# 使用示例
if __name__ == "__main__":
    output_json = "visualization/face_landmarks_features.json"

    prefix = "/data/Leo/mm/data/raw_data/NanfangHospital/cry/wqq-baby-f/output/"
    input_jsons = glob.glob(prefix + "*cam0_face_landmarks.json")

    # prefix = "/data/Leo/mm/data/raw_data/"
    # input_jsons = glob.glob(prefix + "*cam0_face_landmarks.json", recursive=True)

    for input_json in input_jsons:
        features = process_json(input_json, output_json)
        print(f"已处理 {len(features)} 帧，示例输出:")
        print(json.dumps(features[0], indent=2))
    