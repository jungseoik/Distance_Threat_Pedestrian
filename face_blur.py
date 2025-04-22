# face_blur.py

import cv2
from PIL import Image
from realutils.detect import detect_heads

def blur_heads_in_video(video_path: str, output_path: str = 'blurred_output.mp4'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("🎥 블러 처리 시작...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV → PIL 변환
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # detect_heads 호출
        results = detect_heads(pil_img)

        # 바운딩 박스에 블러 처리
        for (x1, y1, x2, y2), label, conf in results:
            head = frame[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(head, (51, 51), 0)
            frame[y1:y2, x1:x2] = blurred

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 저장 완료: {output_path}")

# 예시 실행
if __name__ == "__main__":
    blur_heads_in_video('threat_2.mp4', 'blurred_output2.mp4')


