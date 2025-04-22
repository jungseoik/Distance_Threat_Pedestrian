# preprocess.py

import os
import shutil
import random
from tqdm import tqdm

# ====================== 설정 ============================
DATASET_ROOT = "/home/piawsa6000/nas192/tmp/jsi/PETA dataset"
OUTPUT_ROOT = "prepro_peta"
LABEL_FILENAME = "Label.txt"
SPLITS = [0.8, 0.1, 0.1]  # train, val, test
VALID_EXTS = ["bmp", "jpg", "jpeg", "png"]

# 태그 정의
MALE_TAG = "personalMale"
FEMALE_TAG = "personalFemale"

# ====================== 함수 ============================

def get_gender_label(label_list):
    """성별 태그를 기반으로 gender 클래스 반환"""
    if MALE_TAG in label_list:
        return "Male"
    elif FEMALE_TAG in label_list:
        return "Female"
    return None

# ================== 출력 디렉토리 준비 ==================

for split in ['train', 'val', 'test']:
    for gender in ['Male', 'Female']:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, gender), exist_ok=True)

# ================== 라벨링된 이미지 수집 ==================

all_labeled_images = []
missing_images = []

for folder_name in os.listdir(DATASET_ROOT):
    folder_path = os.path.join(DATASET_ROOT, folder_name)
    archive_path = os.path.join(folder_path, "archive")
    label_path = os.path.join(archive_path, LABEL_FILENAME)

    if not os.path.isdir(archive_path) or not os.path.isfile(label_path):
        continue

    archive_images = os.listdir(archive_path)

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            raw_key = parts[0]
            label_list = parts[1:]

            gender = get_gender_label(label_list)
            if gender is None:
                continue

            # CUHK 스타일 (0001.png) vs 일반 스타일 (3_134_...)
            if '.' in raw_key:
                label_key = raw_key.split('.')[0]  # e.g. 0001
                mode = 'filename'
            else:
                label_key = raw_key              # e.g. 3
                mode = 'index'

            # 실제 이미지 매칭
            if mode == 'filename':
                matched = [
                    img for img in archive_images
                    if img.split('.')[0] == label_key
                ]
            else:
                matched = [
                    img for img in archive_images
                    if img.startswith(f"{label_key}_") and img.split('.')[-1].lower() in VALID_EXTS
                ]

            if not matched:
                missing_images.append((folder_name, raw_key, gender))
                continue

            for img_name in matched:
                all_labeled_images.append({
                    "src_path": os.path.join(archive_path, img_name),
                    "gender": gender,
                    "img_name": f"{folder_name}_{img_name}"  # 중복 방지용
                })

print(f"✅ 총 라벨링된 이미지 수: {len(all_labeled_images)}")

# ================== 데이터 분할 ==================

random.seed(42)
random.shuffle(all_labeled_images)

n_total = len(all_labeled_images)
n_train = int(SPLITS[0] * n_total)
n_val = int(SPLITS[1] * n_total)
n_test = n_total - n_train - n_val  # 잔여는 test로

dataset_splits = {
    'train': all_labeled_images[:n_train],
    'val': all_labeled_images[n_train:n_train + n_val],
    'test': all_labeled_images[n_train + n_val:]
}

print("\n📊 데이터 분할 정보")
print(f"  총 수       : {n_total}")
print(f"  Train       : {n_train}")
print(f"  Validation  : {n_val}")
print(f"  Test        : {n_test}")
print(f"  합계        : {n_train + n_val + n_test}")

# ================== 이미지 복사 ==================

for split_name, images in dataset_splits.items():
    print(f"\n📦 {split_name} 세트: {len(images)}장 복사 중...")
    for item in tqdm(images):
        dst_path = os.path.join(OUTPUT_ROOT, split_name, item["gender"], item["img_name"])
        shutil.copy2(item["src_path"], dst_path)

# ================== 누락 보고 ==================

print(f"\n⚠️ 누락된 라벨(이미지 없음): {len(missing_images)}")
for folder, idx, gender in missing_images[:10]:
    print(f" - {folder} / {idx} ({gender})")
if len(missing_images) > 10:
    print("... (이하 생략)")

print("\n✅ 전처리 완료! YOLOv8 학습 준비 완료.")
