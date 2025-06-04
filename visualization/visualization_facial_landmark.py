import cv2
import json
import numpy as np

def visualize_mouth_points(video_path, json_path, output_img_path):
    """检查第一帧的嘴部关键点标注是否正确"""
    # 1. 加载JSON数据
    with open(json_path) as f:
        data = json.load(f)
    
    # 2. 获取视频第一帧
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    
    # 3. 获取第一帧对应的landmarks数据
    first_frame_landmarks = data['frames'][0]['landmarks']  # 应是一个二维数组
    
    # 4. 定义测试的嘴部点索引（根据您的68点模型顺序）
    # 注意：Python列表索引从0开始，所以标准68点模型的编号需要-1
    all_points = [
        48, 49, 50, 51, 52, 53, 54,  # 上外唇 (7点)
        55, 56, 57, 58, 59,          # 下外唇 (5点)
        60, 61, 62, 63, 64,          # 上内唇 (5点)
        65, 66, 67,                  # 下内唇 (3点)

        
        36, 37, 38, 39, 40, 41,      # 左眼 (6点)
        42, 43, 44, 45, 46, 47       # 右眼 (6点) 
    ]

    
    # 5. 在帧上绘制关键点
    for idx, point in enumerate(first_frame_landmarks):
        if idx in all_points:
            x, y = map(int, point)  # 直接解包坐标数组

            
            # 绘制带编号的点
            cv2.circle(frame, (x, y), 1, (0, 100, 255), -1)

    # 6. 保存结果图片
    cv2.imwrite(output_img_path, frame)
    cap.release()

# 使用示例
visualize_mouth_points(
    video_path="/data/Leo/mm/data/raw_data/NanfangHospital/non-cry/csl-baby-f/csl-baby-f_2025-03-19-18-58-58_cam0.avi",
    json_path="/data/Leo/mm/data/raw_data/NanfangHospital/non-cry/csl-baby-f/output/csl-baby-f_2025-03-19-18-58-58_cam0_face_landmarks.json",
    output_img_path="visualization_facial_landmark.jpg"
)