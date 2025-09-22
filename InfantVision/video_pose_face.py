import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from retinaface import RetinaFace
import lib.models as face_models
from lib.config import config as face_config, update_config
from lib.utils.transforms import crop
from lib.core.evaluation import decode_preds
import glob

class Visualizer:
    def __init__(self):
        self.body_part_colors = {}  # 存储BGR格式颜色
        self.landmark_size = 4
        self.landmark_color = (0, 0, 255)  # BGR红色
        self.mask_blend_weights = (0.7, 0.3)

    def set_colors_rgb(self, color_dict):
        """将RGB格式颜色转换为BGR格式存储"""
        self.body_part_colors = {}
        for part, color in color_dict.items():
            if len(color) == 4:  # RGBA
                self.body_part_colors[part] = (color[2], color[1], color[0], color[3])
            elif len(color) == 3:  # RGB
                self.body_part_colors[part] = (color[2], color[1], color[0], 1.0)

    def draw_landmarks(self, frame, landmarks):
        for point in landmarks[0]:
            x, y = map(int, point)
            cv2.circle(frame, (x, y), self.landmark_size, self.landmark_color, -1)
        return frame

    def draw_body_mask(self, frame, body_mask):
        # 分离通道
        alpha = body_mask[:, :, 3]  # 0-1范围
        
        # ======================= [代码修正处] =======================
        # 原始代码错误地将已经是0-255范围的颜色值又乘以了255，导致颜色信息丢失
        # bgr = (body_mask[:, :, :3] * 255).astype(np.uint8) # 这是错误行
        # 正确做法是直接转换数据类型
        bgr = body_mask[:, :, :3].astype(np.uint8)
        # ==========================================================
        
        # 创建3通道alpha
        alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        
        # 精确混合
        result = frame.copy()
        result = (result * (1 - alpha_3ch) + bgr * alpha_3ch).astype(np.uint8)
        return result

class VideoHumanAnalyzer:
    def __init__(self, config_path='/root/mm/InfantVision/experiments/300w/hrnet-r90jt.yaml',
                 model_path='/data/Leo/mm/models/hrnet-r90jt.pth'):
        self.visualizer = Visualizer()
        self.body_parser = pipeline(Tasks.image_segmentation, 
                                      'iic/cv_resnet101_image-multiple-human-parsing', 
                                      device='cuda')
        self.face_model = self._init_face_model(config_path, model_path)

    def _init_face_model(self, config_path, model_path):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', default=config_path)
        parser.add_argument('--model_file', default=model_path)
        # 使用一个空的列表来避免解析命令行参数
        update_config(face_config, parser.parse_args([])) 
        model = face_models.get_face_alignment_net(face_config).cuda()
        model = nn.DataParallel(model)
        state_dict = torch.load(model_path)
        if 'state_dict' in state_dict:
            model.module.load_state_dict(state_dict['state_dict'])
        else:
            model.module.load_state_dict(state_dict)
        model.eval()
        return model

    def _white_balance(self, image, limit=10.0):
        avg = np.mean(image, axis=(0, 1))
        ratios = np.array([min(max(avg)/c, limit) for c in avg]).reshape(1, 1, 3)
        return np.clip(image * ratios, 0, 255).astype(np.uint8)

    def _parse_body(self, image):
        result = self.body_parser(image)
        h, w = image.shape[:2]
        visual_mask = np.zeros((h, w, 4), dtype=np.float32)
        
        label_map = {
            'Face': 'Head', 'head': 'Head', 'face': 'Head',
            'Left-arm': 'Arms', 'Right-arm': 'Arms', 'left_arm': 'Arms', 'right_arm': 'Arms',
            'Left-upper-arm': 'Arms', 'Right-upper-arm': 'Arms',
            'Left-lower-arm': 'Arms', 'Right-lower-arm': 'Arms',
            'Torso-skin': 'Torso', 'torso': 'Torso', 'upper': 'Torso',
            'Left-leg': 'Legs', 'Right-leg': 'Legs', 'left_leg': 'Legs', 'right_leg': 'Legs',
            'Left-upper-leg': 'Legs', 'Right-upper-leg': 'Legs',
            'Left-lower-leg': 'Legs', 'Right-lower-leg': 'Legs'
        }
        
        for label, mask in zip(result[OutputKeys.LABELS], result['masks']):
            mapped_label = label_map.get(label, None)
            if mapped_label and mapped_label in self.visualizer.body_part_colors:
                visual_mask[mask > 0] = self.visualizer.body_part_colors[mapped_label]
        
        return visual_mask

    def _detect_face_and_landmarks(self, image):
        faces = RetinaFace.detect_faces(image)
        if not isinstance(faces, dict) or not faces:
            return None
        best_face = max(faces.values(), key=lambda x: (x['facial_area'][2] - x['facial_area'][0]) *
                                                (x['facial_area'][3] - x['facial_area'][1]))['facial_area']
        scale = max(best_face[2] - best_face[0], best_face[3] - best_face[1]) / 200 * 1.25
        center = torch.Tensor([(best_face[0] + best_face[2]) / 2, (best_face[1] + best_face[3]) / 2])
        img = np.array(Image.fromarray(image).convert('RGB'), dtype=np.float32)
        img = crop(img, center, scale, face_config.MODEL.IMAGE_SIZE, rot=0)
        img = (img / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = torch.Tensor(img.transpose([2, 0, 1])).unsqueeze(0).cuda()
        with torch.no_grad():
            output = self.face_model(img)
        landmarks = decode_preds(output.data.cpu(), center, scale, [64, 64]).numpy()
        return image, landmarks

    def _process_frame(self, frame):
        balanced = self._white_balance(frame)
        rgb_frame = cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
        body_mask = self._parse_body(rgb_frame)
        result = frame.copy()
        result = self.visualizer.draw_body_mask(result, body_mask)
        face_result = self._detect_face_and_landmarks(rgb_frame)
        if face_result:
            result = self.visualizer.draw_landmarks(result, face_result[1])
        return result, face_result is not None

    def process_video(self, video_path, output_dir, target_frame=None):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.png")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return None
        
        if target_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"无效帧号: {target_frame}")
                return None
            result_frame, has_face = self._process_frame(frame)
            if has_face:
                cv2.imwrite(output_path, result_frame)
                print(f"结果已保存: {output_path}")
            else:
                print(f"警告: 第 {target_frame} 帧未检测到人脸")
            cap.release()
            return output_path if has_face else None
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_idx in tqdm(range(total_frames), desc=f"处理 {os.path.basename(video_path)}"):
                ret, frame = cap.read()
                if not ret:
                    continue
                # 优化：在循环内部只调用一次detect_faces
                detected_faces = RetinaFace.detect_faces(frame)
                if isinstance(detected_faces, dict) and len(detected_faces) > 0:
                    result_frame, _ = self._process_frame(frame)
                    cap.release()
                    cv2.imwrite(output_path, result_frame)
                    print(f"结果已保存: {output_path}")
                    return output_path
            cap.release()
            print(f"未检测到有效帧: {video_path}")
            return None

def main():
    analyzer = VideoHumanAnalyzer()

    # ====== 可调节参数区域 (使用RGB格式) ======
    trans = 0.4  # 透明度 (0=透明, 1=不透明)
    
    # 关键点设置
    analyzer.visualizer.landmark_size = 4
    analyzer.visualizer.landmark_color = (255, 255, 255)
    
    # 身体部位颜色 (RGB格式)
    analyzer.visualizer.set_colors_rgb({
        'Head': (50, 205, 50, trans),    # RGB绿色
        'Arms': (186, 85, 255, trans),   # RGB紫色
        'Torso': (255, 191, 0, trans),   # RGB金色
        'Legs': (220, 20, 60, trans)     # RGB红色
    })
    
    # 混合权重 (原图权重, 掩码权重) - 注意: 此参数在当前代码中未被使用
    analyzer.visualizer.mask_blend_weights = (0.7, 0.3)
    # ====== 参数区域结束 ======
    input_dir = "/data/Leo/mm/data/ShenzhenUniversityGeneralHospital/data/"
    # input_dir = "/data/Leo/mm/data/NanfangHospital/data/"
    # output_dir = "/data/Leo/mm/output/"
    output_dir = './'

    target_frame = None  # 指定处理的帧号
        
    # 处理找到的第一个视频 (可以修改索引[0:1]来处理不同视频)
    for video_path in glob.glob(os.path.join(input_dir, "*.avi"))[1:2]: 
        video_path = os.path.join(input_dir, 'zj-baby-f_2025-08-02-16-17-18.avi')
        analyzer.process_video(video_path, output_dir, target_frame)

if __name__ == "__main__":
    main()