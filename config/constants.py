CLS_MODEL_PATH = "Gender_cls/artifacts/weights/best.pt"
DET_MODEL_PATH = "models/PersonDet_v3.2.0.pt"
DISTANCE_THRESHOLD = 300
SAMPLE_VIDEO_PATH = "assets/threat_1.mp4"
INPUT_VIDEO_PATH = "assets/threat_1.mp4"
OUTPUT_VIDEO_PATH = "output/result.mp4"

# Person Detection
DET_REPO_ID = "PIA-SPACE-LAB/PersonDet_v3.2.0"
DET_MODEL_NAME = "PersonDet_v3.2.0.pt"

# Gender Classification
CLS_REPO_ID = "PIA-SPACE-LAB/PersonGenderCls_v3.2.0"
CLS_MODEL_NAME = "PersonGenderCls_v3.2.0.pt"

# Data preprocess
AIHUB_DATASET_ROOT = "/home/piawsa6000/nas192/tmp/jsi/015.한국인재식별이미지/01.데이터"
AIHUB_OUTPUT_DIR = "prepro_data/prepro_aihub"

PETA_DATASET_ROOT = "assets/PETA dataset"
PETA_OUTPUT_DIR = "prepro_data/prepro_peta"
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# peta data prepro 태그 정의
MALE_TAG = "personalMale"
FEMALE_TAG = "personalFemale"
LABEL_FILENAME = "Label.txt"
SPLITS = [0.8, 0.1, 0.1]  # train, val, test