import os
import warnings
import logging
import shutil


# === 1. 彻底关闭所有 Python 警告 ===
warnings.filterwarnings("ignore")  # 全局忽略所有警告

# === 2. 禁用所有 logging 输出 ===
logging.disable(logging.CRITICAL)  # 关闭所有 logging（包括 INFO/WARNING）

# === 4. 设置 TensorFlow 环境变量（必须在 import 之前）===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 3=ERROR, 2=WARNING, 1=INFO, 0=DEBUG
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 禁用 oneDNN 的浮点顺序警告

import numpy as np
import cv2
import matplotlib.pyplot as plt

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import glob
from tqdm import tqdm
import json


def delete_files_in_directory(directory):    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # 删除文件
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


def delete_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"成功删除文件夹: {folder_path}")
            return True
        else:
            print(f"文件夹不存在: {folder_path}")
            return False
    except Exception as e:
        print(f"删除文件夹时出错: {folder_path} - 错误: {e}")
        return False
    

def simple_white_balance_with_gain_limit(image, limit=10.0):
    # 计算每个通道的平均值
    avg_b, avg_g, avg_r = np.mean(image, axis=(0, 1))

    # 计算最大平均值
    max_avg = max(avg_b, avg_g, avg_r)

    # 计算调整比例
    blue_ratio = min(max_avg / avg_b, limit)
    green_ratio = min(max_avg / avg_g, limit)
    red_ratio = min(max_avg / avg_r, limit)

    # 应用白平衡
    wb_image = image.copy()
    wb_image[:, :, 0] = np.clip(wb_image[:, :, 0] * blue_ratio, 0, 255)
    wb_image[:, :, 1] = np.clip(wb_image[:, :, 1] * green_ratio, 0, 255)
    wb_image[:, :, 2] = np.clip(wb_image[:, :, 2] * red_ratio, 0, 255)

    return wb_image.astype(np.uint8)


def save_img(balanced_frame, visual_mask, output_dir, video_name_without_ext):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # 显示原始图像
    axs[0].imshow(balanced_frame)
    axs[0].set_title('Image')
    axs[0].axis('off')

    # 显示带掩码的图像
    axs[1].imshow(balanced_frame)
    axs[1].imshow(visual_mask, alpha=0.5)  # 透明度设置为0.5
    axs[1].set_title('Image with Mask')
    axs[1].axis('off')

    axs[2].imshow(visual_mask, cmap='gray')
    axs[2].set_title('Mask Only')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{video_name_without_ext}_seg_result.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_visual_mask(frame, part_masks_dict, body_part_colors):
    """
    生成可视化掩码图像
    参数:
        frame: 原始帧图像
        part_masks_dict: 各部位的分割掩码字典 {部位名: [mask1, mask2...]}
        body_part_colors: 各部位颜色配置 {部位名: (R,G,B,A)}
    返回:
        visual_mask: 可视化掩码 (RGBA格式)
        blended_frame: 原始帧与掩码的混合结果
    """
    height, width = frame.shape[:2]
    
    # 创建RGBA格式的可视化掩码
    visual_mask = np.zeros((height, width, 4), dtype=np.float32)
    visual_mask[..., 3] = 1.0  # Alpha通道
    
    # 为每个部位填充颜色
    for part_name, masks in part_masks_dict.items():
        color = body_part_colors[part_name]
        for mask in masks:
            visual_mask[mask > 0] = color
    
    # 混合原始帧和掩码 (50%透明度)
    blended_frame = cv2.addWeighted(
        frame, 0.5, 
        visual_mask[:, :, :3].astype(np.uint8), 
        0.5, 0
    )
    
    return visual_mask, blended_frame


def Generate_Intime_Mask_AVI(input_video_path, segmentation_pipeline, body_part_colors):
    # 创建输出目录
    # output_dir = os.path.join(os.path.dirname(input_video_path), 'body')
    # 创建输出目录，在上一层目录下
    output_dir = os.path.join(os.path.dirname(os.path.dirname(input_video_path)), 'Body')
    os.makedirs(output_dir, exist_ok=True)

    # 读取视频文件
    video_capture = cv2.VideoCapture(input_video_path)
    print(f"Processing video: {input_video_path}")
    
    # 获取视频信息
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # 准备JSON输出数据结构
    motion_analysis_results = {
        "video_info": {
            "path": input_video_path,
            "fps": float(fps),
            "frame_count": total_frames,
            "resolution": [float(width), float(height)],
        },
        "features": []
    }
    
    # 创建高质量视频写入对象
    video_name_without_ext = os.path.splitext(os.path.basename(input_video_path))[0]
    output_json_path = os.path.join(output_dir, f"{video_name_without_ext}_motion_features.json")
    output_video_path = os.path.join(output_dir, f"{video_name_without_ext}_masked.avi")
    video_codec = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(output_video_path, video_codec, fps, (width, height))    

    # # 检查输出文件是否已存在
    # if (os.path.exists(output_video_path) and 
    #         os.path.exists(output_json_path)):
    #         existing_pngs = [f for f in os.listdir(output_dir) 
    #                         if f.startswith(f"{video_name_without_ext}_seg_result") and f.endswith('.png')]
    #         if existing_pngs:
    #             print("Output files already exist. Skipping processing.")
    #             return

    # 进度条初始化
    # total_frames = 120
    progress_bar = tqdm(total=total_frames, desc="Processing Video", unit='frame')
    
    current_frame_index = 0
    previous_gray_frame = None
    visual_mask = np.zeros((height, width, 4), dtype=np.float32)
    while True:
    # while current_frame_index <= total_frames:
        ret, current_frame = video_capture.read()
        if not ret:
            break
        
        # 白平衡处理
        frame_copy = current_frame.copy()
        balanced_frame = simple_white_balance_with_gain_limit(frame_copy)
        
        # 处理分割掩码
        part_masks_dict = {part_name: [] for part_name in body_part_colors.keys()}
        part_scores_dict = {part_name: [] for part_name in body_part_colors.keys()}        
        segmentation_result = segmentation_pipeline(balanced_frame)
        for label, mask, score in zip(segmentation_result[OutputKeys.LABELS], segmentation_result['masks'], segmentation_result["scores"]):
            if label in body_part_colors.keys():
                part_masks_dict[label].append(mask)
                part_scores_dict[label].append(score)

        # 生成可视化结果
        visual_mask, blended_frame = generate_visual_mask(
            balanced_frame, 
            part_masks_dict, 
            body_part_colors
        )
        
        # 存储第一帧的分割结果
        if current_frame_index == 0:
            save_img(balanced_frame, visual_mask, output_dir, video_name_without_ext)

        # 计算光流
        gray_frame = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2GRAY)
        optical_flow = None
        if previous_gray_frame is not None:
            optical_flow = cv2.calcOpticalFlowFarneback(previous_gray_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        previous_gray_frame = gray_frame

        if "Face" in part_masks_dict and len(part_masks_dict["Face"]) > 0:
            face_mask = part_masks_dict["Face"][0]  # 取第一个面部mask
            
            # 找到非零像素的行列坐标
            rows = np.any(face_mask, axis=1)  # 高度方向
            cols = np.any(face_mask, axis=0)  # 宽度方向
            
            if np.any(rows) and np.any(cols):
                # 计算边界坐标
                y_min, y_max = np.where(rows)[0][[0, -1]]  # 高度边界
                x_min, x_max = np.where(cols)[0][[0, -1]]  # 宽度边界

                cv2.rectangle(blended_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255))
                # 写入标记视频
                video_writer.write(blended_frame)

                head_height = y_max - y_min  # 实际高度（像素）
                head_width = x_max - x_min   # 实际宽度（像素）
            else:
                head_width = head_height = 0  # 无效mask
        else:
            head_width = head_height = 0  # 未检测到面部

        # 准备当前帧的特征数据
        frame_motion_features = {
            "Frame": current_frame_index,
            "head": [float(head_width), float(head_height)],
            "Face": [],
            "Left-arm": [],
            "Right-arm": [],
            "Left-leg": [],
            "Right-leg": [],
            "Torso-skin": [],
            "WholeFrameMotion": [],
        }
        
        # 计算各部位的运动强度
        if optical_flow is not None:
            for part_name, masks in part_masks_dict.items():
                for mask_idx, mask in enumerate(masks):
                    # 计算该部位mask区域的光流强度
                    if mask.sum() > 0:  # 确保mask有效
                        mask_indices = mask > 0
                        # 计算平均位移
                        shift_x, shift_y = np.mean(optical_flow[mask_indices], axis=0)
                        # 计算位移幅度shift_r
                        shift_r = np.sqrt(shift_x ** 2 + shift_y ** 2)
                        # 计算位移角度（转换为角度制）
                        shift_a = np.degrees(np.arctan2(shift_y, shift_x))
                        # 获取对应的分割分数
                        current_score = part_scores_dict[part_name][mask_idx]
                        # 存储完整运动特征向量
                        motion_vector = [
                            float(shift_x), 
                            float(shift_y), 
                            float(shift_r), 
                            float(shift_a),
                            float(current_score)
                        ]
                        frame_motion_features[part_name].append(motion_vector)

        # WholeFrameMotion改为与其他部位一致的结构
        if optical_flow is not None:
            shift_x, shift_y = np.mean(optical_flow, axis=(0,1))
            shift_r = np.sqrt(shift_x**2 + shift_y**2)
            shift_a = np.degrees(np.arctan2(shift_y, shift_x))
            frame_motion_features["WholeFrameMotion"] = [[
                float(shift_x), float(shift_y), 
                float(shift_r), float(shift_a)
            ]]

        motion_analysis_results["features"].append(frame_motion_features)        

        current_frame_index += 1
        progress_bar.update(1)  # 更新进度条

    # 保存JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(motion_analysis_results, f, indent=2)

    # 释放资源
    video_capture.release()
    video_writer.release()
    progress_bar.close()
    
    print(f"\nProcessing completed. Results saved in: {output_dir}")


def main(): 
    prefix = "/data/Leo/mm/data/raw_data/"
    # video_files = glob.glob(prefix + "*/*/*/*.avi")
    # files = [
    #     # "NanfangHospital/cry/drz-m",
    #     # "ShenzhenUniversityGeneralHospital/non-cry/zlj-baby-f",
    #     "NanfangHospital/cry/llj-baby-f",
    #     # "NanfangHospital/non-cry/lyz-m",
    # ]
    # dirs = [prefix + file + "/*0.avi" for file in files]
    # video_files = [glob.glob(d) for d in dirs]
    # video_files = [item for sublist in video_files for item in sublist]

    prefix = '/data/Leo/mm/data/Newborn200/data/'
    video_files = glob.glob(prefix + '*.mp4')
    for video_file in video_files:
        Generate_Intime_Mask_AVI(
            input_video_path=video_file,
            segmentation_pipeline=pipeline(Tasks.image_segmentation, 'iic/cv_resnet101_image-multiple-human-parsing', device='cuda:1'),
            body_part_colors={
                'Left-arm': (255, 0, 0, 1),      # Red for left arm
                'Right-arm': (255, 0, 0, 1),     # Red for right arm
                'Left-leg': (0, 0, 255, 1),      # Blue for left leg
                'Right-leg': (0, 0, 255, 1),     # Blue for right leg
                'Face': (255, 255, 255, 1),      # White for face
                'Torso-skin': (0, 255, 0, 1),    # Green for torso
            },
        )




if __name__ == "__main__":
    main()