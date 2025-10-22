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
from datetime import timedelta
import argparse

class Visualizer:
    def __init__(self):
        self.body_colors = {}
        self.landmark_radius = 4
        self.landmark_color = (0, 0, 255)
        self.mask_blend_weights = (0.7, 0.3)

    def set_colors_hex(self, color_dict):
        """Convert HEX color strings to BGR format with optional alpha."""
        self.body_colors = {}
        for part, hex_info in color_dict.items():
            hex_str, alpha = hex_info if isinstance(hex_info, tuple) else (hex_info, 1.0)
            r = int(hex_str[1:3], 16)
            g = int(hex_str[3:5], 16)
            b = int(hex_str[5:7], 16)
            self.body_colors[part] = (b, g, r, alpha)

    def draw_landmarks(self, frame, landmarks):
        """Draw circular landmarks on the frame."""
        for x, y in landmarks[0]:
            cv2.circle(frame, (int(x), int(y)), self.landmark_radius, self.landmark_color, -1)
        return frame

    def draw_body_mask(self, frame, body_mask):
        """Apply body mask to frame with alpha blending."""
        alpha = body_mask[:, :, 3]
        bgr = body_mask[:, :, :3].astype(np.uint8)
        alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        return (frame * (1 - alpha_3ch) + bgr * alpha_3ch).astype(np.uint8)

    def format_timestamp(self, seconds):
        """Format timestamp into HH:MM:SS.mmm string."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

class VideoHumanAnalyzer:
    def __init__(self, config_path='/root/mm/InfantVision/experiments/300w/hrnet-r90jt.yaml',
                 model_path='/data/Leo/mm/models/hrnet-r90jt.pth'):
        self.visualizer = Visualizer()
        self.body_parser = pipeline(Tasks.image_segmentation, 
                                  'iic/cv_resnet101_image-multiple-human-parsing', 
                                  device='cuda')
        self.face_model = self._init_face_model(config_path, model_path)

    def _init_face_model(self, config_path, model_path):
        """Initialize face alignment model with pre-trained weights."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', default=config_path)
        parser.add_argument('--model_file', default=model_path)
        update_config(face_config, parser.parse_args([]))
        model = face_models.get_face_alignment_net(face_config).cuda()
        model = nn.DataParallel(model)
        state_dict = torch.load(model_path)
        model.module.load_state_dict(state_dict.get('state_dict', state_dict))
        model.eval()
        return model

    def _white_balance(self, image, limit=10.0):
        """Apply white balance to image with clipping limit."""
        avg_color = np.mean(image, axis=(0, 1))
        ratios = np.clip(avg_color / avg_color, 1/limit, limit).reshape(1, 1, 3)
        return np.clip(image * ratios, 0, 255).astype(np.uint8)

    def _parse_body(self, image):
        """Parse body parts and create a colored mask."""
        result = self.body_parser(image)
        height, width = image.shape[:2]
        mask = np.zeros((height, width, 4), dtype=np.float32)

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

        for label, seg_mask in zip(result[OutputKeys.LABELS], result['masks']):
            part = label_map.get(label)
            if part in self.visualizer.body_colors:
                mask[seg_mask > 0] = self.visualizer.body_colors[part]
        return mask

    def _detect_face_and_landmarks(self, image):
        """Detect face and landmarks using RetinaFace and face model."""
        faces = RetinaFace.detect_faces(image)
        if not isinstance(faces, dict) or not faces:
            return None, 0
        best_face = max(faces.values(), key=lambda x: (x['facial_area'][2] - x['facial_area'][0]) *
                                              (x['facial_area'][3] - x['facial_area'][1]))
        facial_area_size = (best_face['facial_area'][2] - best_face['facial_area'][0]) * \
                           (best_face['facial_area'][3] - best_face['facial_area'][1])
        best_face_area = best_face['facial_area']
        scale = max(best_face_area[2] - best_face_area[0], best_face_area[3] - best_face_area[1]) / 200 * 1.25
        center = torch.Tensor([(best_face_area[0] + best_face_area[2]) / 2, (best_face_area[1] + best_face_area[3]) / 2])
        img = np.array(Image.fromarray(image).convert('RGB'), dtype=np.float32)
        img = crop(img, center, scale, face_config.MODEL.IMAGE_SIZE, rot=0)
        img = (img / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = torch.Tensor(img.transpose([2, 0, 1])).unsqueeze(0).cuda()
        with torch.no_grad():
            output = self.face_model(img)
        landmarks = decode_preds(output.data.cpu(), center, scale, [64, 64]).numpy()
        return image, landmarks, facial_area_size

    def _process_frame(self, frame, frame_index=None, timestamp=None, idx=None):
        """Process a single frame: white balance, body parsing, and landmark detection."""
        balanced_frame = self._white_balance(frame)
        rgb_frame = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2RGB)
        body_mask = self._parse_body(rgb_frame)
        result_frame = self.visualizer.draw_body_mask(frame, body_mask)
        face_data = self._detect_face_and_landmarks(rgb_frame)
        has_face = face_data[0] is not None
        if has_face:
            result_frame = self.visualizer.draw_landmarks(result_frame, face_data[1])
        
        # Add specified index number in white at the top-left corner
        if idx is not None:
            # Convert idx to string and get text size
            text = str(idx)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)
            # Set y coordinate with a larger offset for better spacing and distance from edge
            y_offset = max(30, text_height + 25)  # Increased offset for aesthetic gap and distance
            cv2.putText(result_frame, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
            
        if frame_index is not None and timestamp is not None:
            time_str = self.visualizer.format_timestamp(timestamp)
            print(f"Frame {frame_index}: Time {time_str}")
            
        return result_frame, has_face, face_data[2] if has_face else 0

    def process_video(self, video_path, output_dir, target_frame=None, idx=None):
        """Process video or specific frame, save output if faces detected."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.png")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if target_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"Invalid frame number: {target_frame}")
                return None
            timestamp = target_frame / fps
            result_frame, has_face, _ = self._process_frame(frame, target_frame, timestamp, idx)
            cap.release()
            if has_face:
                cv2.imwrite(output_path, result_frame)
                print(f"Saved result to: {output_path}")
                print(f"Frame {target_frame}/{total_frames}: Time {self.visualizer.format_timestamp(timestamp)}")
                return output_path
            print(f"Warning: No face detected in frame {target_frame}")
            return None

        for frame_index in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                continue
            timestamp = frame_index / fps
            faces = RetinaFace.detect_faces(frame)
            if isinstance(faces, dict) and faces:
                result_frame, _, _ = self._process_frame(frame, frame_index, timestamp, idx)
                cap.release()
                cv2.imwrite(output_path, result_frame)
                print(f"Saved result to: {output_path}")
                print(f"Frame {frame_index}/{total_frames}: Time {self.visualizer.format_timestamp(timestamp)}")
                return output_path
        cap.release()
        print(f"No valid frames detected in: {video_path}")
        return None

    def process_frame_range(self, video_path, output_dir, start_frame, num_frames=30, sample_mode='s'):
            """
            Process frames starting from start_frame, with two sampling modes:
            - 'per_second': sample one frame per second (total num_frames seconds)
            - 'continuous': sample consecutive frames (total num_frames frames)
            
            Save frames with faces to output_dir, return best frame number based on largest body mask area.
            """
            os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.splitext(os.path.basename(video_path))[0]

            best_frame_num = None
            best_body_area = 0
            frame_results = []

            # Determine step based on sampling mode
            if sample_mode == 's':
                step = int(fps)  # One frame per second
                total_to_process = min(num_frames, (total_frames - start_frame) // step + 1)
                desc = f"Processing {num_frames} seconds for {basename}"
            elif sample_mode == 'c':
                step = 1  # Consecutive frames
                total_to_process = min(num_frames, total_frames - start_frame)
                desc = f"Processing {num_frames} frames for {basename}"
            else:
                raise ValueError("sample_mode must be 'per_second' or 'continuous'")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for i in tqdm(range(total_to_process), desc=desc):
                frame_num = start_frame + i * step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = frame_num / fps
                
                # Compute body mask and area
                balanced_frame = self._white_balance(frame)
                rgb_frame = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2RGB)
                body_mask = self._parse_body(rgb_frame)
                body_area = np.sum(body_mask[:, :, 3] > 0)
                
                # Detect face
                face_data = self._detect_face_and_landmarks(rgb_frame)
                has_face = face_data[0] is not None
                
                # Create result frame
                result_frame = self.visualizer.draw_body_mask(frame, body_mask)
                if has_face:
                    result_frame = self.visualizer.draw_landmarks(result_frame, face_data[1])
                
                if has_face:
                    # Add frame number annotation
                    cv2.putText(result_frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    output_path = os.path.join(output_dir, f"{basename}_frame_{frame_num:06d}.png")
                    cv2.imwrite(output_path, result_frame)
                    frame_results.append((frame_num, body_area, timestamp))
                    if body_area > best_body_area:
                        best_body_area = body_area
                        best_frame_num = frame_num
                    print(f"Saved frame {frame_num} to {output_path}: Body area {body_area}")
                else:
                    print(f"Frame {frame_num}: No face detected")

            cap.release()

            if frame_results:
                print(f"\nBest frame in range: {best_frame_num} (body area: {best_body_area})")
                for fn, ba, ts in sorted(frame_results, key=lambda x: x[1], reverse=True)[:5]:
                    time_str = self.visualizer.format_timestamp(ts)
                    print(f"Top frames: {fn} (area: {ba}, time: {time_str})")
            else:
                print("No frames with faces detected in the range.")

            return best_frame_num

def main():
    trans = 0.4
    analyzer = VideoHumanAnalyzer()
    analyzer.visualizer.landmark_radius = 4
    analyzer.visualizer.landmark_color = (255, 255, 255)
    analyzer.visualizer.set_colors_hex({
        'Head': ('#E6B89C', trans),   # 肉色
        'Torso': ('#66B2FF', trans),  # 蓝色
        'Arms': ('#99E599', trans),   # 浅绿色
        'Legs': ('#CC99FF', trans)    # 浅紫色
    })
    analyzer.visualizer.mask_blend_weights = (0.7, 0.3)

    input_dir = "/data/Leo/mm/data/NICU50/data/"
    infants = [
        ("zyd-baby-m_2025-08-07-17-02-59", 3960), # 1好，保温箱，无眼罩，哭
        ("qjj-baby-m_2025-07-23-20-32-42", 960),  # 2好，保温箱，有眼罩，哭
        ("ljx-baby-m_2025-07-28-17-33-32", 1826), # 3好，保温箱，无眼罩，安静
        ("qj-baby-f_2025-07-28-18-29-10", 9455),  # 4好，保温箱，有眼罩，安静
        ("hbxd-m_2025-07-29-15-06-59", 3163),     # 5好，保温箱，身体遮挡，侧脸，安静
        ("wxl-baby-m_2025-08-05-12-19-40", 1431), # 6差，保温箱，戴眼罩，侧脸，哭
        ("xgx-baby-m_2025-07-29-15-39-53", 7230), # 7差，摇篮，身体遮挡，嘴部遮挡，安静      
        ("ysqd-f_2025-07-29-15-32-40", 4590),     # 8差，摇篮，身体遮挡，嘴部遮挡，哭 
    ]

    # # Example: Test frame ranges first
    # for infant, start_frame in infants:
    #     video_path = os.path.join(input_dir, f"{infant}.avi")
    #     analyzer.process_frame_range(video_path, f'./img/{infant}', start_frame, num_frames=240, sample_mode='c')
    
    # Then process best frames with original method
    for idx, (infant, target_frame) in enumerate(infants):
        video_path = os.path.join(input_dir, f"{infant}.avi")
        analyzer.process_video(video_path, './img', target_frame)

if __name__ == "__main__":
    main()