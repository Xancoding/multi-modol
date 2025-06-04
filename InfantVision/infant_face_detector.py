import cv2
import torch
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device

class InfantFaceDetector:
    def __init__(self, weights_path, img_size=640, conf_thres=0.02, iou_thres=0.5, device=''):
        set_logging()
        self.device = select_device(device)
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load model
        self.model = attempt_load(weights_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        
    def dynamic_resize(self, shape, stride=64):
        max_size = max(shape[0], shape[1])
        if max_size % stride != 0:
            max_size = (int(max_size / stride) + 1) * stride 
        return max_size
    
    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]
        coords[:, :10] /= gain
        
        coords[:, 0].clamp_(0, img0_shape[1])
        coords[:, 1].clamp_(0, img0_shape[0])
        coords[:, 2].clamp_(0, img0_shape[1])
        coords[:, 3].clamp_(0, img0_shape[0])
        coords[:, 4].clamp_(0, img0_shape[1])
        coords[:, 5].clamp_(0, img0_shape[0])
        coords[:, 6].clamp_(0, img0_shape[1])
        coords[:, 7].clamp_(0, img0_shape[0])
        coords[:, 8].clamp_(0, img0_shape[1])
        coords[:, 9].clamp_(0, img0_shape[0])
        return coords
    
    def calculate_orientation_angle(self, landmarks, img_width):
        """Calculate face orientation angle based on facial landmarks"""
        p0 = [landmarks[0], landmarks[1]]  # left eye
        p1 = [landmarks[2], landmarks[3]]  # right eye
        q = [landmarks[4], landmarks[5]]   # nose
        
        a = np.array([[-q[0]*(p1[0]-p0[0]) - q[1]*(p1[1]-p0[1])], 
                     [-p0[1]*(p1[0]-p0[0]) + p0[0]*(p1[1]-p0[1])]])
        b = np.array([[p1[0] - p0[0], p1[1] - p0[1]], 
                     [p0[1] - p1[1], p1[0] - p0[0]]])
        
        if not (np.all(a == 0) or np.all(b == 0)):
            ProjPoint = -1 * np.matmul(np.linalg.inv(b), a)
            proj_point = np.round(ProjPoint.transpose())
            u = np.array([proj_point[0][0], proj_point[0][1]]) - np.array(q)
            v = np.array([img_width, q[1]]) - np.array(q)
            val = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
            Theta = np.arccos(np.amin(np.array([np.amax(val, -1), 1])))
            ThetaInDegrees = np.rad2deg(Theta)
            vc = np.cross(np.append(u, 0), np.append(v, 0))
            angle = ThetaInDegrees if vc[2] > 0 else 360 - ThetaInDegrees
            return angle
        return None
    
    def detect_face(self, img0):
        """Detect faces in the image and return the largest face bounding box"""
        # Determine image size
        imgsz = self.img_size
        if imgsz <= 0:
            imgsz = self.dynamic_resize(img0.shape)
        imgsz = check_img_size(imgsz, s=self.stride)
        
        # Preprocess image
        img = letterbox(img0, imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]
        
        # Find the largest face
        max_area = 0
        best_face = None
        if pred is not None:
            for det in pred:
                # Scale coordinates back to original image size
                det = det.unsqueeze(0)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = det[0, :4].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    best_face = [int(x1), int(y1), int(x2), int(y2)]
        
        return best_face
    
    def detect(self, img0):
        """Detect faces with landmarks and orientation"""
        # Determine image size
        imgsz = self.img_size
        if imgsz <= 0:
            imgsz = self.dynamic_resize(img0.shape)
        imgsz = check_img_size(imgsz, s=self.stride)
        
        # Preprocess image
        img = letterbox(img0, imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]
        
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(self.device)
        gn_lks = torch.tensor(img0.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(self.device)
        boxes = []
        h, w, c = img0.shape
        
        if pred is not None:
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
            pred[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], pred[:, 5:15], img0.shape).round()
            
            # Find the face with highest confidence
            if len(pred) > 0:
                max_conf_idx = torch.argmax(pred[:, 4])
                j = max_conf_idx.item()
                
                xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
                xywh = xywh.data.cpu().numpy()
                conf = pred[j, 4].cpu().numpy()
                landmarks = (pred[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = pred[j, 15].cpu().numpy()
                
                # Calculate orientation angle
                angle = self.calculate_orientation_angle(landmarks, w)
                
                x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
                y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
                x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
                y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
                boxes.append([x1, y1, x2-x1, y2-y1, conf])
                
                # Draw results
                img0 = self.show_results(img0, xywh, conf, landmarks, class_num, angle)
                
        return img0, boxes
    
    def show_results(self, img, xywh, conf, landmarks, class_num, angle=None):
        h, w, c = img.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1
        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

        # Draw landmarks
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for i in range(5):
            point_x = int(landmarks[2 * i] * w)
            point_y = int(landmarks[2 * i + 1] * h)
            cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

        # Draw confidence and angle
        tf = max(tl - 1, 1)
        label = f'Conf: {conf:.2f}'
        if angle is not None:
            label += f' | Angle: {angle:.1f}°'
        
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img


def process_video(video_path, output_path, weights_path):
    detector = InfantFaceDetector(weights_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, boxes = detector.detect(frame)
        
        if processed_frame is not None:
            cv2.imwrite(output_path, processed_frame)        
    
    cap.release()

def process_img(img_path, output_path, weights_path):
    detector = InfantFaceDetector(weights_path)
    
    frame = cv2.imread(img_path)
    processed_frame, boxes = detector.detect(frame)
    
    if processed_frame is not None:
        cv2.imwrite(output_path, processed_frame)
        print(f"检测结果已保存到: {output_path}")
    else:
        print("未检测到人脸")


if __name__ == '__main__':
    import argparse
    import glob
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/data/Leo/mm/models/nicuface_y5f.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.02, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    
    # Example usage
    OUTPUT_PATH = "infant_face_detector.png"
    WEIGHTS_PATH = opt.weights[0] if isinstance(opt.weights, list) else opt.weights
    
    # Process video
    prefix = "/data/Leo/mm/data/raw_data/NanfangHospital/cry/wqq-baby-f/"
    video_files = glob.glob(prefix + "*cam0.avi")
    for video_file in video_files:
        process_video(video_file, OUTPUT_PATH, WEIGHTS_PATH)