import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import glob
import cv2
import json
from tqdm import tqdm
from retinaface import RetinaFace
from PIL import Image
import numpy as np

import lib.models as models
from lib.config import config, update_config
from lib.utils.transforms import crop
from lib.core.evaluation import decode_preds
from infant_face_orientation import FaceOrientationEstimator

class FaceAlignmentProcessor:
    def __init__(self, config_path, model_path):
        self.args = self._parse_args(config_path, model_path)
        self.model = self._initialize_model()
        self.face_detector = FaceOrientationEstimator('/data/Leo/mm/models/face.pt')
        # self.face_detector = InfantFaceDetector('/data/Leo/mm/models/nicuface_y5f.pt')
        
    @staticmethod
    def _parse_args(config_path, model_path):
        parser = argparse.ArgumentParser(description='Train Face Alignment')
        parser.add_argument('--cfg', default=config_path, 
                          help='experiment configuration filename', type=str)
        parser.add_argument('--model_file', help='model parameters',
                          default=model_path, type=str)
        args = parser.parse_args([])  # Empty list to avoid actual command line parsing
        update_config(config, args)
        return args
    
    def _initialize_model(self):
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.determinstic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

        config.defrost()
        config.MODEL.INIT_WEIGHTS = False
        config.freeze()
        
        model = models.get_face_alignment_net(config)
        gpus = [config.GPUS] if isinstance(config.GPUS, int) else list(config.GPUS)
        model = nn.DataParallel(model, device_ids=gpus).cuda()
        
        state_dict = torch.load(self.args.model_file)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict)
        else:
            model.module.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def detect_face(self, image):
        """检测人脸并返回最大的人脸框"""
        faces = RetinaFace.detect_faces(image)
        
        if isinstance(faces, dict):
            max_area = 0
            best_face = None
            for face in faces.values():
                facial_area = face['facial_area']
                area = (facial_area[2] - facial_area[0]) * (facial_area[3] - facial_area[1])
                if area > max_area:
                    max_area = area
                    best_face = facial_area
            return best_face
        return None
    
    def prepare_input(self, image, bbox, image_size):
        """准备模型输入数据"""
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
        center = torch.Tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        scale *= 1.25
        
        img = np.array(image.convert('RGB'), dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img = crop(img, center, scale, image_size, rot=0)
        img = (img / 255.0 - mean) / std
        img = torch.Tensor(img.transpose([2, 0, 1])).unsqueeze(0)
        
        return img, center, scale
    
    def predict_landmarks(self, image, bbox):
        """预测面部关键点"""
        inp, center, scale = self.prepare_input(image, bbox, config.MODEL.IMAGE_SIZE)
        
        with torch.no_grad():
            output = self.model(inp)
        
        score_map = output.data.cpu()
        preds = decode_preds(score_map, center, scale, [64, 64])
        return preds.numpy()
    
    def draw_results(self, image, preds):
        """在图像上绘制关键点和边界框"""
        # 绘制关键点
        for point in preds[0, :, :]:
            cv2.circle(image, tuple(map(int, point)), 2, (255, 255, 0), 1)

        # 绘制边界框
        x_min, y_min = map(int, np.min(preds[0, :, :], axis=0))
        x_max, y_max = map(int, np.max(preds[0, :, :], axis=0))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return image
    
    def process_frame(self, frame):
        """处理单帧图像"""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        try:
            face_bbox = self.face_detector.detect_face(frame)
            if face_bbox is None:
                return frame, None
            
            preds = self.predict_landmarks(pil_image, face_bbox)
            frame = self.draw_results(frame, preds)
            
            # 将关键点转换为列表格式
            landmarks = preds[0].tolist()  # shape (68, 2)
            return frame, landmarks
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame, None
    
    def process_image(self, image_path, output_path):
        """处理单张图像"""
        cv_image = cv2.imread(image_path)
        result, landmarks = self.process_frame(cv_image)
        if landmarks is not None:
            with open(output_path.replace('.png', '.json'), 'w') as f:
                json.dump({'landmarks': landmarks}, f)
        cv2.imwrite(output_path, result)
        print(f"Result saved to {output_path}")
    
    def process_video(self, video_path):
        """处理视频文件，保存处理后的视频和关键点JSON文件"""
        # 创建输出目录
        # output_dir = os.path.join(os.path.dirname(video_path), 'face')
        output_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'Face')
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置输出文件路径
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{base_name}_face_landmarks.avi")
        output_json_path = os.path.join(output_dir, f"{base_name}_face_landmarks.json")
        # output_img_path 现在会在第一次检测到人脸时动态生成

        # 检查输出文件是否已存在
        if (os.path.exists(output_video_path) and 
            os.path.exists(output_json_path)):
            existing_pngs = [f for f in os.listdir(output_dir) 
                            if f.startswith(f"{base_name}_face_landmarks_frame") and f.endswith('.png')]
            if existing_pngs:
                print(f"\n输出文件已存在，跳过处理: {os.path.basename(video_path)}")
                return {
                    'video_path': output_video_path,
                    'json_path': output_json_path,
                    'image_path': os.path.join(output_dir, existing_pngs[0]),  # 返回第一个找到的PNG
                    'status': 'skipped'
                }
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 初始化结果字典
        results = {
            'video_info': {
                'path': video_path,
                'fps': fps,
                'frame_count': total_frames,
                'resolution': f"{width}x{height}"
            },
            'frames': []
        }
        
        # 处理每一帧
        first_frame_saved = False
        output_img_path = None  # 初始化为None，将在第一次检测到人脸时设置
        for frame_idx in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, landmarks = self.process_frame(frame)
            
            # 写入处理后的视频帧
            out.write(processed_frame)
            
            # 保存结果
            frame_data = {
                'frame_number': frame_idx,
                'timestamp': frame_idx / fps,
                'landmarks': landmarks if landmarks is not None else None
            }
            results['frames'].append(frame_data)

            if not first_frame_saved and landmarks is not None:
                # 在文件名中包含帧号
                output_img_path = os.path.join(output_dir, f"{base_name}_face_landmarks.png")
                cv2.imwrite(output_img_path, processed_frame)
                print(f"\n首次检测到人脸 - 帧号: {frame_idx}, 保存为: {os.path.basename(output_img_path)}")
                first_frame_saved = True
        
        # 释放资源
        cap.release()
        out.release()
        
        # 保存JSON结果
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessing completed for {os.path.basename(video_path)}")
        return {
            'video_path': output_video_path,
            'json_path': output_json_path,
            'image_path': output_img_path,  # 如果检测到人脸则包含路径，否则为None
            'status': 'completed'
        }

def main():
    # 配置路径
    config_path = '/root/mm/InfantVision/experiments/300w/hrnet-r90jt.yaml'
    model_path = '/data/Leo/mm/models/hrnet-r90jt.pth'
    
    # 初始化处理器
    processor = FaceAlignmentProcessor(config_path, model_path)

    # # dataset Newborn200
    # prefix = '/data/Leo/mm/data/Newborn200/data/'
    # # video_files = [prefix + x + '.mp4' for x in ['01', '02', '03', '04', '05']]
    # video_files = glob.glob(prefix + "*.mp4")

    # dataset NICU
    # prefix = "/data/Leo/mm/data/ShenzhenUniversityGeneralHospital/data/"
    # prefix = "/data/Leo/mm/data/NanfangHospital/data/"
    prefix = "/data/Leo/mm/data/badNICU50/data/"
    video_files = glob.glob(prefix + "*.avi")

    for video_file in video_files:
        processor.process_video(video_file)


if __name__ == '__main__':
    main()