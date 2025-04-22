# Distance_Threat_Pedestrian


<div align="center">
  <img src="https://pds.saramin.co.kr/company/logo/202208/26/rh7mst_fm5y-1purslx_logo.png" width="100" alt="Hugging Face Logo" />
</div>

<div align="center">
  <a href="https://huggingface.co/PIA-SPACE-LAB/PersonDet_v3.2.0">
    <img alt="Person Detection Model"
         src="https://img.shields.io/badge/🤗 View on Hugging Face-PersonDet_v3.2.0-yellow?style=flat-square" />
  </a>
  <a href="https://huggingface.co/PIA-SPACE-LAB/PersonGenderCls_v3.2.0">
        <img alt="Gender Classification Model"
            src="https://img.shields.io/badge/🤗 View on Hugging Face-GenderCls_v3.2.0-yellow?style=flat-square" />
  </a>
</div>


사람을 탐지하고, 성별을 분류하며, 사람 간 거리를 기반으로 **위협 상황을 판단**하는 영상 분석 시스템입니다.  
이 프로젝트는 `YOLOv11` 기반 커스텀 모델을 사용하며, Streamlit을 통해 **직관적인 웹 UI**를 제공합니다.

---

## 설치 방법

```bash
# Conda 환경 생성
conda create -n gender python=3.10 -y
conda activate gender

# 패키지 설치
pip install -r requirements.txt

# Ultralytics 커스텀 YOLOv11 버전 설치
pip uninstall ultralytics
git clone https://github.com/jyrainer/ultralytics.git
cd ultralytics
git checkout new_exp_yolov11-8.3.66
pip install -e .
```

> ⚠️ 본 레포는 `YOLOv11` 전용입니다.  
> `YOLOv8`, `YOLOv9`, `YOLOv10` 버전에서는 작동하지 않습니다.

---

## 🚀 사용 방법

###  1. Streamlit 웹 앱 실행

```bash
conda activate gender
python -m streamlit run app.py
```

### ▶️ Streamlit에서 제공하는 기능

- 🎞️ 업로드한 영상의 프레임 실시간 표시
- 👥 탐지된 사람 개별 크롭 이미지 표시
- 🚨 거리 위협 발생 시 붉은 게이지와 경고 메시지 표시

---

## 2. Python 코드에서 추론 사용 (`main.py`)

`ThreatVideoDiscriminator` 클래스를 통해 영상 파일을 직접 처리할 수 있습니다.

```python
from models.threat import ThreatVideoDiscriminator

# 1️⃣ 성별 분류 포함 (남/여 예측 + 거리 위협 판단)
model = ThreatVideoDiscriminator(use_classifier=True, output_path="output/result1.mp4")
model.process_video("assets/threat_1.mp4")

# 2️⃣ 성별 분류 제외 (사람 탐지 + 거리 위협 판단만)
model = ThreatVideoDiscriminator(use_classifier=False, output_path="output/result2.mp4")
model.process_video("assets/threat_1.mp4")
```

> `use_classifier=True` 설정 시 성별 분류기가 작동하며, `"Male"` / `"Female"` 라벨이 출력됩니다.

---

## 3. 성별 분류기 학습 (`train.py`)

학습은 YOLO 분류기 형식으로 진행됩니다.

```python
# train.py
from ultralytics import YOLO

model = YOLO("Gender_cls/ai_hub2/weights/last.pt")
model.train(cfg="config/gender_cls_aihub.yaml")
```

### 학습 실행법

```bash
python train.py
```

### 구성 예시

```
config/
├── gender_cls_aihub.yaml  # 학습 설정
Gender_cls/
├── ai_hub2/
│   └── weights/           # 가중치 저장 위치
```

---

## config/gender_cls_aihub.yaml 예시

```yaml
task: classify
model: yolov8m-cls.pt
data: Gender_cls/data/aihub.yaml
epochs: 300
imgsz: 224
batch: 64
workers: 8
```

---

## 구성된 주요 모듈

| 모듈명 | 설명 |
|--------|------|
| `PersonDetector` | YOLOv11 기반 사람 탐지기 |
| `GenderClassifier` | 성별 분류기 (여자: 0, 남자: 1) |
| `ThreatAnalyzer` | 사람 간 거리 계산 및 위협 시각화 |
| `ThreatVideoDiscriminator` | 위 세 기능을 통합한 추론 파이프라인 |
| `app.py` | Streamlit 기반 웹 분석 UI |
| `train.py` | 성별 분류기 학습 코드 |

---

## 📌 참고 사항

- 모델 가중치는 실행 시 자동 다운로드되며, `/models/`, `/Gender_cls/` 내부에 저장됩니다.
- 거리 기준은 `config.py` 내 `DISTANCE_THRESHOLD` 값을 통해 조정 가능합니다.

---

