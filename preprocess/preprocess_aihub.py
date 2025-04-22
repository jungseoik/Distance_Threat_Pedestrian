import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 설정
DATASET_ROOT = "/home/piawsa6000/nas192/tmp/jsi/015.한국인재식별이미지/01.데이터"
OUTPUT_DIR = "prepro_aihub"

SPLITS = {
    "train": {
        "img_dir": os.path.join(DATASET_ROOT, "1.Training", "원천_Training"),
        "xml_dir": os.path.join(DATASET_ROOT, "1.Training", "라벨_Training"),
    },
    "val": {
        "img_dir": os.path.join(DATASET_ROOT, "2.Validation", "원천_Validation"),
        "xml_dir": os.path.join(DATASET_ROOT, "2.Validation", "라벨_Validation"),
    },
}

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# 출력 폴더 생성
for split in SPLITS:
    for gender in ["Male", "Female"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, gender), exist_ok=True)

def extract_gender(xml_path):
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

# 누락 로그 저장
skipped_files = []

for split_name, paths in SPLITS.items():
    print(f"\n📦 처리 중: {split_name}")
    img_files = [f for f in os.listdir(paths["img_dir"]) if os.path.splitext(f)[1].lower() in VALID_EXT]

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

        dst_path = os.path.join(OUTPUT_DIR, split_name, gender, img_file)
        shutil.copy2(img_path, dst_path)

# 결과 요약
print("\n✅ 전처리 완료! YOLO 성별 분류 학습 데이터셋 구성 완료.")

if skipped_files:
    print(f"\n⚠️ 성별 태그 누락 또는 오류로 제외된 파일 수: {len(skipped_files)}")
    for i, (file, reason) in enumerate(skipped_files[:10], start=1):
        print(f"  {i:02d}. {file}: {reason}")
    if len(skipped_files) > 10:
        print("  ... (이하 생략)")
