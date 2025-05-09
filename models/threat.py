import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from utils.prepro import preprocess_v2, load_model
import config

class PersonDetector:
    def __init__(self, model_path: str = config.DET_MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        boxes, centers = [], []
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                boxes.append((x1, y1, x2, y2))
                centers.append((cx, cy))
        return boxes, np.array(centers)


class GenderClassifier:
    def __init__(self, model_path: str = config.CLS_MODEL_PATH, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path)
        self.fp16 = self.model.fp16

    def classify(self, crop_img):
        img = preprocess_v2(crop_img, device=self.device, shape=[224, 224], scaleup=True)
        img = img if isinstance(img, torch.Tensor) else torch.from_numpy(img).to(self.device)
        img = img.half() if self.fp16 else img.float()
        result = self.model(img)[0]
        return result.argmax().item()


class ThreatAnalyzer:
    def __init__(self, distance_threshold: int = config.DISTANCE_THRESHOLD):
        self.distance_threshold = distance_threshold

    def draw(self, frame, boxes, genders=None, centers=None):
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if genders:
                gender = genders[i]
                label = "Female" if gender == 0 else "Male"
                color = (255, 0, 255) if gender == 0 else (0, 255, 255)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                color = (255, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if centers is not None and len(centers) >= 2:
            dist_matrix = cdist(centers, centers, metric='euclidean')
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    if dist_matrix[i][j] < self.distance_threshold:
                        p1, p2 = tuple(centers[i]), tuple(centers[j])
                        cv2.line(frame, p1, p2, (0, 0, 255), 2)
                        cv2.putText(frame, f"{int(dist_matrix[i][j])}px", p1,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame


class ThreatVideoDiscriminator:
    def __init__(self,
                 use_classifier=True,
                 det_model_path: str = config.DET_MODEL_PATH,
                 cls_model_path: str = config.CLS_MODEL_PATH,
                 output_path: str = config.OUTPUT_VIDEO_PATH):

        self.use_classifier = use_classifier
        self.output_path = output_path

        self.detector = PersonDetector(det_model_path)
        self.classifier = GenderClassifier(cls_model_path) if use_classifier else None
        self.analyzer = ThreatAnalyzer(distance_threshold=config.DISTANCE_THRESHOLD)

    def process_frame(self, frame):
        boxes, centers = self.detector.detect(frame)
        genders = []
        if self.use_classifier:
            for (x1, y1, x2, y2) in boxes:
                crop = frame[y1:y2, x1:x2]
                gender = self.classifier.classify(crop)
                genders.append(gender)
        else:
            genders = None

        return self.analyzer.draw(frame, boxes, genders, centers)

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated = self.process_frame(frame)
            video_writer.write(annotated)

        cap.release()
        video_writer.release()
        print(f"🎬 저장 완료 → {self.output_path}")
