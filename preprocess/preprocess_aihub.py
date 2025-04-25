import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import config
from utils.custom_logger import custom_logger

# 로거 생성
logger = custom_logger(__name__)

def process_aihub_dataset():
    """
    AI Hub 데이터셋을 처리하여 YOLO 성별 분류 학습 데이터셋으로 구성합니다.
    
    - 원본 데이터셋에서 성별 정보를 추출하여 Male/Female 폴더로 이미지를 분류합니다.
    - XML 파일에서 성별 정보를 추출하고, 누락되거나 잘못된 정보의 파일은 건너뜁니다.
    - 처리 결과를 로그로 출력합니다.
    """
    SPLITS = {
        "train": {
            "img_dir": os.path.join(config.AIHUB_DATASET_ROOT, "1.Training", "원천_Training"),
            "xml_dir": os.path.join(config.AIHUB_DATASET_ROOT, "1.Training", "라벨_Training"),
        },
        "val": {
            "img_dir": os.path.join(config.AIHUB_DATASET_ROOT, "2.Validation", "원천_Validation"),
            "xml_dir": os.path.join(config.AIHUB_DATASET_ROOT, "2.Validation", "라벨_Validation"),
        },
    }

    # 출력 폴더 생성
    for split in SPLITS:
        for gender in ["Male", "Female"]:
            os.makedirs(os.path.join(config.AIHUB_OUTPUT_DIR, split, gender), exist_ok=True)

    # 누락 로그 저장
    skipped_files = []

    for split_name, paths in SPLITS.items():
        logger.info(f"\n📦 처리 중: {split_name}")
        img_files = [f for f in os.listdir(paths["img_dir"]) if os.path.splitext(f)[1].lower() in config.VALID_EXT]

        for img_file in tqdm(img_files):
            xml_file = os.path.splitext(img_file)[0] + ".xml"
            xml_path = os.path.join(paths["xml_dir"], xml_file)
            img_path = os.path.join(paths["img_dir"], img_file)

            if not os.path.exists(xml_path):
                skipped_files.append((img_file, "❌ XML 파일 없음"))
                continue

            gender, reason = extract_gender(xml_path)
            if gender not in {"Male", "Female"}:
                skipped_files.append((img_file, reason))
                continue

            dst_path = os.path.join(config.AIHUB_OUTPUT_DIR, split_name, gender, img_file)
            shutil.copy2(img_path, dst_path)

    # 결과 요약
    logger.info("\n✅ 전처리 완료! YOLO 성별 분류 학습 데이터셋 구성 완료.")

    if skipped_files:
        logger.warning(f"\n⚠️ 성별 태그 누락 또는 오류로 제외된 파일 수: {len(skipped_files)}")
        for i, (file, reason) in enumerate(skipped_files[:10], start=1):
            logger.warning(f"  {i:02d}. {file}: {reason}")
        if len(skipped_files) > 10:
            logger.warning("  ... (이하 생략)")

def extract_gender(xml_path):
    """
    XML 파일에서 성별 정보를 추출합니다.
    
    Args:
        xml_path (str): XML 파일 경로
        
    Returns:
        tuple: (성별값, 오류메시지) 형태로 반환
            - 성공 시: ("Male"|"Female", None)
            - 실패 시: (None, 오류 메시지)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        gender_tag = root.find(".//gender")
        if gender_tag is None:
            return None, "❌ <gender> 태그 없음"
        if not gender_tag.text or gender_tag.text.strip() == "":
            return None, "⚠️ <gender> 태그 비어 있음"
        g = gender_tag.text.strip().lower()
        if g == "male":
            return "Male", None
        elif g == "female":
            return "Female", None
        return None, f"⚠️ 정의되지 않은 gender 값: '{g}'"
    except ET.ParseError:
        return None, "❌ XML 파싱 오류"
    except Exception as e:
        return None, f"❗ 기타 예외: {e}"
