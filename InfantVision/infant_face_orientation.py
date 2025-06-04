import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression_face, scale_coords, xyxy2xywh
from utils.datasets import letterbox



class FaceOrientationEstimator:
    def __init__(self, weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = attempt_load(weights_path, map_location=device)
        self.img_size = 800
        self.conf_thres = 0.02
        self.iou_thres = 0.5
        
    def process_frame(self, frame):
        # Preprocess
        img0 = frame.copy()
        h0, w0 = frame.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        pred = self.model(img)[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)
        
        # Process detections (only take the most confident one)
        detections = []
        angle = None
        if len(pred[0]) > 0:
            # Sort by confidence and take the highest one
            det = pred[0]
            det = det[det[:, 4].argsort(descending=True)]
            det = det[:1]  # Only keep the most confident detection
            
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            det[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()
            
            # Get face detection
            xywh = (xyxy2xywh(det[0, :4].view(1, 4)) / torch.tensor(frame.shape)[[1, 0, 1, 0]].to(self.device)).view(-1).tolist()
            landmarks = (det[0, 5:15].view(1, 10) / torch.tensor(frame.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(self.device)).view(-1).tolist()
            conf = det[0, 4].item()
            
            # Prepare detections output
            h, w = frame.shape[:2]
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            
            land = []
            for i in range(5):
                land.append(int(landmarks[2*i] * w))
                land.append(int(landmarks[2*i+1] * h))
                
            detections = [x1, y1, x2-x1, y2-y1, conf] + land
            
            # Calculate orientation angle
            p0 = [land[0], land[1]]  # left eye
            p1 = [land[2], land[3]]   # right eye
            q = [land[4], land[5]]    # nose
            a = np.array([[-q[0]*(p1[0]-p0[0]) - q[1]*(p1[1]-p0[1])], 
                         [-p0[1]*(p1[0]-p0[0]) + p0[0]*(p1[1]-p0[1])]])
            b = np.array([[p1[0] - p0[0], p1[1] - p0[1]], 
                         [p0[1] - p1[1], p1[0] - p0[0]]])
            
            if not (np.all(a == 0) or np.all(b == 0)):
                ProjPoint = -1 * np.matmul(np.linalg.inv(b), a)
                proj_point = np.round(ProjPoint.transpose())
                u = np.array([proj_point[0][0], proj_point[0][1]]) - np.array(q)
                v = np.array([w, q[1]]) - np.array(q)
                val = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
                Theta = np.arccos(np.amin(np.array([np.amax(val, -1), 1])))
                ThetaInDegrees = np.rad2deg(Theta)
                vc = np.cross(np.append(u, 0), np.append(v, 0))
                angle = ThetaInDegrees if vc[2] > 0 else 360 - ThetaInDegrees
                
            # Draw results on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            for i in range(5):
                cv2.circle(frame, (land[2*i], land[2*i+1]), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"Angle: {angle:.1f}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return frame, detections, angle
    
    def scale_coords_landmarks(self, img1_shape, coords, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        
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

    def detect_face(self, image):
        """Detect faces in the image and return the largest face bounding box"""
        # Process frame through the model
        img0 = image.copy()
        h0, w0 = image.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        pred = self.model(img)[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)
        
        # Find the largest face
        max_area = 0
        best_face = None
        if len(pred[0]) > 0:
            for det in pred[0]:
                # Scale coordinates back to original image size
                det = det.unsqueeze(0)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = det[0, :4].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    best_face = [int(x1), int(y1), int(x2), int(y2)]
        
        if best_face is not None:
            return best_face
        else:
            raise ValueError("No face detected in the image")
        

def process_video(video_path, output_path, weights_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
        
    estimator = FaceOrientationEstimator(weights_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame (only one face)
        processed_frame, detections, angle = estimator.process_frame(frame)
        
        cv2.imwrite(output_path, processed_frame)
                    
            
    cap.release()

def process_img(img_path, output_path, weights_path):
    estimator = FaceOrientationEstimator(weights_path)

    frame = cv2.imread(img_path)
    processed_frame, detections, angle = estimator.process_frame(frame)
    
    if processed_frame is not None:
        cv2.imwrite(output_path, processed_frame)
        print(f"检测结果已保存到: {output_path}")
    else:
        print("未检测到人脸")



if __name__ == '__main__':
    # 配置参数
    VIDEO_FILE =  "/root/mm/data/raw_data/NanfangHospital/cry/drz-m/drz-m_2025-03-19-11-32-18_cam0.avi" # 识别，但不稳定
    # VIDEO_FILE = "/root/mm/data/raw_data/NanfangHospital/cry/zzy-baby-m/zzy-baby-m_2025-03-19-15-08-19_cam0.avi"  # 识别
    # VIDEO_FILE =  "/root/mm/data/raw_data/NanfangHospital/cry/wpy-baby-f/wpy-baby-f_2025-03-19-17-33-48_cam0.avi" # 识别，偶尔不准
    # VIDEO_FILE =  "/root/mm/data/raw_data/NanfangHospital/cry/wqq-baby-f/wqq-baby-f_2025-03-19-17-18-21_cam0.avi" # 识别
    # VIDEO_FILE =  "/root/mm/data/raw_data/NanfangHospital/cry/wxy-baby-m/wxy-baby-m_2025-03-19-14-58-47_cam0.avi" # 未识别，侧脸侧的太夸张，太难了，考虑不可能检测到

    OUTPUT_PATH = "infant_face_orientation.png"
    WEIGHTS_PATH = "/data/Leo/mm/models/face.pt"
    
    # 处理视频
    process_video(VIDEO_FILE, OUTPUT_PATH, WEIGHTS_PATH)

    # IMG_FILE = "/root/mm/code/InfAnFace/hxy.png"
    # process_img(IMG_FILE, OUTPUT_PATH, WEIGHTS_PATH)