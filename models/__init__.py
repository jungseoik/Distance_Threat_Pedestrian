import os
from huggingface_hub import hf_hub_download
import config

MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

# 탐지 모델 다운로드
det_path = os.path.join(MODEL_DIR, config.DET_MODEL_NAME)
if not os.path.exists(det_path):
    try:
        print(f"🔄 Downloading {config.DET_MODEL_NAME} from {config.DET_REPO_ID}...")
        hf_hub_download(
            repo_id=config.DET_REPO_ID,
            filename=config.DET_MODEL_NAME,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Downloaded: {config.DET_MODEL_NAME}")
    except Exception as e:
        print(f"❌ Error downloading detection model: {e}")
else:
    print(f"✅ Detection model already exists at: {det_path}")

# 분류 모델 다운로드
cls_path = os.path.join(MODEL_DIR, config.CLS_MODEL_NAME)
if not os.path.exists(cls_path):
    try:
        print(f"🔄 Downloading {config.CLS_MODEL_NAME} from {config.CLS_REPO_ID}...")
        hf_hub_download(
            repo_id=config.CLS_REPO_ID,
            filename=config.CLS_MODEL_NAME,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Downloaded: {config.CLS_MODEL_NAME}")
    except Exception as e:
        print(f"❌ Error downloading classification model: {e}")
else:
    print(f"✅ Classification model already exists at: {cls_path}")
