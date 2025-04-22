import cv2  # OpenCV: 영상 처리 및 시각화 라이브러리
import numpy as np  # NumPy: 수치 계산 및 배열 처리
import os  # OS 관련 기능 (폴더 생성 등)
from ultralytics import YOLO  # YOLOv8 모델 로딩 라이브러리
from scipy.spatial.distance import cdist  # scipy 거리 행렬 함수 (cdist = "cross distance")
import models

# 🚨 거리 임계값 설정 (픽셀 단위) → 이 값보다 가까우면 거리 위반으로 판단
DISTANCE_THRESHOLD = 300

# 🎯 YOLOv8 사전 학습 모델 불러오기
model = YOLO("models/PersonDet_v3.2.0.pt")  

# 🎥 VideoCapture 객체 생성 (줄여서 cap이라 부름)
# cap = capture의 약자. 프레임 단위로 영상을 읽는 객체
cap = cv2.VideoCapture("blurred_output.mp4")

# 💾 영상 저장 준비
# fourcc = four character code (비디오 코덱 식별자)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 형식용 코덱 지정
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # 영상 프레임 속도 (초당 프레임 수)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 프레임 가로 너비
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 세로 높이

# 출력 폴더 생성 (이미 있으면 무시)
os.makedirs("output", exist_ok=True)

# 🎬 비디오 저장 객체 생성
# out = 출력 비디오 객체 (VideoWriter)
out = cv2.VideoWriter("output/result.mp4", fourcc, fps, (width, height))

# 🎞️ 영상 프레임 처리 루프 시작
while cap.isOpened():
    ret, frame = cap.read()  # ret: 성공 여부 (True/False), frame: 현재 프레임 이미지
    if not ret:
        break  # 영상 끝났거나 오류 시 루프 종료

    results = model(frame)[0]  # YOLOv8으로 탐지 실행 (첫 번째 결과 가져옴)
    centers = []  # 사람 중심 좌표들 저장 리스트 (cx, cy)
    boxes = []    # 바운딩 박스 좌표 저장 리스트 [(x1, y1, x2, y2)]

    # 👤 사람만 필터링 (COCO class ID 0 = person)
    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 바운딩 박스 좌표
            cx = int((x1 + x2) / 2)  # 중심점 x
            cy = int((y1 + y2) / 2)  # 중심점 y
            centers.append((cx, cy))  # 중심점 저장
            boxes.append((int(x1), int(y1), int(x2), int(y2)))  # 박스 저장

    centers = np.array(centers)  # 거리 계산을 위해 numpy 배열로 변환

    # ✅ 모든 사람에게 초록 박스 먼저 그림
    for box in boxes:
        cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 2)  # 초록 박스

    # 📏 사람 간 거리 계산 (N x N 거리 행렬)
    if len(centers) >= 2:
        # cdist: center들 간의 유클리디안 거리 계산
        dist_matrix = cdist(centers, centers, metric='euclidean')

        # 🔴 거리 위반 시각화: 가까운 쌍만 빨간 선/거리 표시
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):  # 중복 방지: (i < j)
                if dist_matrix[i][j] < DISTANCE_THRESHOLD:
                    p1 = tuple(centers[i].astype(int))  # 첫 번째 사람 위치
                    p2 = tuple(centers[j].astype(int))  # 두 번째 사람 위치
                    cv2.line(frame, p1, p2, (0, 0, 255), 2)  # 빨간 선 그리기
                    cv2.putText(frame, f"{int(dist_matrix[i][j])}px", p1,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 거리 숫자 출력

    # 📝 프레임 저장 (output/result.mp4에 기록됨)
    out.write(frame)

# 🧹 자원 정리
cap.release()  # 캡처 종료
out.release()  # 비디오 저장 종료
print("🎬 영상 저장 완료! → output/result.mp4")
