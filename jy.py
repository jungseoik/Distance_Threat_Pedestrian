import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from utils.prepro import box_iou, xywh2xyxy, load_model, LetterBox, preprocess_v2, non_max_suppression

model = YOLO("models/PersonDet_v3.2.0.pt")
gender_model = load_model("/home/piawsa6000/work/seoik/Classfication_gender_pedestrian/Gender_cls/ai_hub/weights/best.pt")  # Using load_model instead of YOLO
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISTANCE_THRESHOLD = 300
SAMPLE_VIDEO_PATH = "assets/threat_1.mp4"

cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
os.makedirs("output", exist_ok=True)
out = cv2.VideoWriter("output/result.mp4", fourcc, fps, (width, height))
transform = LetterBox(scaleup=False, auto=False)  # Add this from the first code

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    centers = []
    boxes = []
    
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # Person class
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centers.append((cx, cy))
            boxes.append((x1, y1, x2, y2))
    
    centers = np.array(centers)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        person_crop = frame[y1:y2, x1:x2]
        
        # Replace this section with the classification approach from the first code
        # ------- Start of replacement -------
        # Preprocess cropped frame for classification
        img = preprocess_v2(person_crop, device=DEVICE, shape=[224, 224], scaleup=True)
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(DEVICE)
        img = img.half() if gender_model.fp16 else img.float()
        
        # Get classification result
        result = gender_model(img)[0]
        pred_class = result.argmax().item()
        # ------- End of replacement -------
        
        label = "Female" if pred_class == 0 else "Male"
        color = (255, 0, 255) if pred_class == 0 else (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if len(centers) >= 2:
        dist_matrix = cdist(centers, centers, metric='euclidean')
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if dist_matrix[i][j] < DISTANCE_THRESHOLD:
                    p1 = tuple(centers[i])
                    p2 = tuple(centers[j])
                    cv2.line(frame, p1, p2, (0, 0, 255), 2)
                    cv2.putText(frame, f"{int(dist_matrix[i][j])}px", p1,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
print("🎬 성별 정보 포함 영상 저장 완료! → output/result.mp4")