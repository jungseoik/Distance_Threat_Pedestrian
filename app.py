import streamlit as st
import tempfile
import time
import cv2
from models.threat import ThreatVideoDiscriminator
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Threat Analysis", layout="wide")
st.title("🎥 Threat Video Analyzer")

video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv", "MOV"])
analyze_button = st.button("🔍 Run Analysis")

col1, col2, col3 = st.columns(3)
if video_file and analyze_button:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(video_file.read())

    processor = ThreatVideoDiscriminator()
    cap = cv2.VideoCapture(temp_video.name)

    with col1:
        st.subheader("🎞️ Live Video Feed")
        video_feed = st.empty()

    with col2:
        st.subheader("👥 Cropped Persons")
        cropped_images_area = st.empty()

    with col3:
        st.subheader("🚨 Alerts")
        gauge_placeholder = st.empty()
        message_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes, centers = processor.detector.detect(frame)
        genders = []
        cropped_imgs = []

        if processor.use_classifier:
            for (x1, y1, x2, y2) in boxes:
                crop = frame[y1:y2, x1:x2]
                cropped_imgs.append(crop)
                gender = processor.classifier.classify(crop)
                genders.append(gender)
        else:
            genders = None

        annotated = processor.analyzer.draw(frame, boxes, genders, centers)

        # col1 - 실시간 프레임 표시
        video_feed.image(annotated, channels="BGR")
        if cropped_imgs:
            # Resize & Convert to RGB
            cropped_imgs_fixed = [
                cv2.cvtColor(cv2.resize(img, (150, 200)), cv2.COLOR_BGR2RGB)
                for img in cropped_imgs
            ]
            cropped_images_area.image(cropped_imgs_fixed, use_container_width=False)

        # col3 - 경고 여부 판단 및 게이지 표시
        alert = False
        alert_level = 0 

        if centers is not None and len(centers) >= 2:
            dist_matrix = cdist(centers, centers, metric='euclidean')
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    if dist_matrix[i][j] < processor.analyzer.distance_threshold:
                        alert = True
                        alert_level = 100
                        break

        if alert:
            gauge_placeholder.progress(alert_level, text="🚨 위험: 사람이 너무 가깝습니다!")
            message_placeholder.markdown("#### 🚨 **거리 임계값을 초과했습니다**")
        else:
            gauge_placeholder.progress(alert_level, text="✅ 정상 상태")
            message_placeholder.markdown("✅ 현재 거리 안전.")

        time.sleep(1 / 30)  # FPS 제한

    cap.release()
    st.success("✅ Analysis completed!")
