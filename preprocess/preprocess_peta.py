import os
import shutil
import random
from tqdm import tqdm
import config
from utils.custom_logger import custom_logger
import re

logger = custom_logger(__name__)

def get_gender_label(label_list):
    """성별 태그를 기반으로 gender 클래스 반환"""
    if config.MALE_TAG in label_list:
        return "Male"
    elif config.FEMALE_TAG in label_list:
        return "Female"
    return None

def process_peta_dataset(debug=True):
    """
    PETA 데이터셋을 처리하여 성별 분류 학습용 데이터셋을 생성합니다.
    
    Args:
        debug (bool): 상세 디버깅 정보 출력 여부
        
    Returns:
        dict: 처리 결과 통계
    """
    # ================== 출력 디렉토리 준비 ==================
    for split in ['train', 'val', 'test']:
        for gender in ['Male', 'Female']:
            os.makedirs(os.path.join(config.PETA_OUTPUT_DIR, split, gender), exist_ok=True)

    # ================== 라벨링된 이미지 수집 ==================
    all_labeled_images = []
    missing_images = []
    dataset_stats = {
        'total_labels': 0,
        'total_images': 0,
        'matched_images': 0,
        'missing_images': 0,
        'folders_processed': 0,
        'folders_skipped': 0,
        'matches_by_mode': {'filename': 0, 'index': 0, 'regex': 0},
        'folder_stats': {}
    }

    # 데이터셋 루트에 있는 모든 폴더 목록
    all_folders = os.listdir(config.PETA_DATASET_ROOT)
    logger.info(f"PETA 데이터셋 루트에서 {len(all_folders)}개 폴더 발견")
    
    for folder_name in all_folders:
        folder_path = os.path.join(config.PETA_DATASET_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        archive_path = os.path.join(folder_path, "archive")
        label_path = os.path.join(archive_path, config.LABEL_FILENAME)

        # 폴더별 통계 초기화
        folder_stats = {
            'label_count': 0,
            'image_count': 0, 
            'matched_count': 0,
            'missing_count': 0
        }

        if not os.path.isdir(archive_path):
            logger.warning(f"폴더 '{folder_name}'에 archive 디렉토리가 없음")
            dataset_stats['folders_skipped'] += 1
            continue
            
        if not os.path.isfile(label_path):
            logger.warning(f"폴더 '{folder_name}'에 라벨 파일({config.LABEL_FILENAME})이 없음")
            dataset_stats['folders_skipped'] += 1
            continue

        dataset_stats['folders_processed'] += 1
        
        # 이미지 파일 목록 (확장자 대소문자 구분 없이)
        archive_files = os.listdir(archive_path)
        archive_images = [f for f in archive_files if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        folder_stats['image_count'] = len(archive_images)
        
        if debug:
            logger.debug(f"폴더 '{folder_name}': {len(archive_images)}개 이미지 파일 발견")
        
        dataset_stats['total_images'] += len(archive_images)

        # 라벨 파일 처리
        with open(label_path, "r") as f:
            label_lines = f.readlines()
            folder_stats['label_count'] = len(label_lines)
            dataset_stats['total_labels'] += len(label_lines)
            
            if debug:
                logger.debug(f"폴더 '{folder_name}': {len(label_lines)}개 라벨 발견")
        
        for line in label_lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            raw_key = parts[0]
            label_list = parts[1:]

            gender = get_gender_label(label_list)
            if gender is None:
                continue
            
            # 여러 매칭 방식 시도
            matched = []
            match_mode = None
            
            # 1. CUHK 스타일 (0001.png 또는 0001) - 파일명 직접 매칭
            if '.' in raw_key:  # 확장자가 있는 경우
                base_key = raw_key.split('.')[0]  # e.g. 0001
                matched = [img for img in archive_images if img.startswith(base_key)]
                if matched:
                    match_mode = 'filename'
                    dataset_stats['matches_by_mode']['filename'] += len(matched)
            
            # 2. 일반 스타일 (숫자, e.g. 3) - 인덱스 매칭
            if not matched and raw_key.isdigit():
                pattern = f"^{raw_key}_"  # e.g. "3_"
                matched = [
                    img for img in archive_images
                    if re.match(pattern, img)
                ]
                if matched:
                    match_mode = 'index'
                    dataset_stats['matches_by_mode']['index'] += len(matched)
            
            # 3. 더 유연한 정규식 매칭 시도
            if not matched:
                # 여러 가능한 패턴 시도
                patterns = [
                    f"^{raw_key}[_-]",  # 시작하는 패턴 (e.g. "3-", "3_")
                    f"[_-]{raw_key}[_-]",  # 중간에 있는 패턴 (e.g. "_3_", "-3-")
                    f"[_-]{raw_key}$"   # 끝나는 패턴 (e.g. "_3", "-3")
                ]
                
                for pattern in patterns:
                    potential_matches = [
                        img for img in archive_images
                        if re.search(pattern, os.path.splitext(img)[0])
                    ]
                    if potential_matches:
                        matched.extend(potential_matches)
                        match_mode = 'regex'
                        dataset_stats['matches_by_mode']['regex'] += len(potential_matches)
                        break

            if not matched:
                missing_images.append((folder_name, raw_key, gender))
                folder_stats['missing_count'] += 1
                dataset_stats['missing_images'] += 1
                
                if debug and folder_stats['missing_count'] <= 5:  # 폴더당 최대 5개까지만 출력
                    logger.debug(f"매칭 실패: 폴더={folder_name}, 라벨키={raw_key}, 성별={gender}")
                continue
            
            folder_stats['matched_count'] += len(matched)
            dataset_stats['matched_images'] += len(matched)
            
            for img_name in matched:
                all_labeled_images.append({
                    "src_path": os.path.join(archive_path, img_name),
                    "gender": gender,
                    "img_name": f"{folder_name}_{img_name}",  # 중복 방지용
                    "match_mode": match_mode
                })

        # 폴더별 통계 저장
        dataset_stats['folder_stats'][folder_name] = folder_stats
        
        # 폴더별 요약 로그
        logger.info(f"폴더 '{folder_name}' 처리 결과: " 
                   f"라벨 {folder_stats['label_count']}개, "
                   f"이미지 {folder_stats['image_count']}개, "
                   f"매칭 {folder_stats['matched_count']}개, "
                   f"미매칭 {folder_stats['missing_count']}개")

    # ================== 매칭 결과 요약 ==================
    logger.info("\n📊 데이터셋 처리 통계")
    logger.info(f"처리된 폴더 수: {dataset_stats['folders_processed']} (건너뛴 폴더: {dataset_stats['folders_skipped']})")
    logger.info(f"총 라벨 수: {dataset_stats['total_labels']}")
    logger.info(f"총 이미지 수: {dataset_stats['total_images']}")
    logger.info(f"매칭된 이미지 수: {dataset_stats['matched_images']}")
    logger.info(f"누락된 라벨 수: {dataset_stats['missing_images']}")
    logger.info(f"매칭 방식별 건수: 파일명={dataset_stats['matches_by_mode']['filename']}, " 
               f"인덱스={dataset_stats['matches_by_mode']['index']}, "
               f"정규식={dataset_stats['matches_by_mode']['regex']}")
    
    # 가장 매칭이 안 된 폴더 상위 5개 출력
    folder_missing_rates = {
        folder: stats['missing_count'] / stats['label_count'] if stats['label_count'] > 0 else 0
        for folder, stats in dataset_stats['folder_stats'].items()
    }
    
    problematic_folders = sorted(
        [(folder, stats['missing_count'], stats['label_count']) 
         for folder, stats in dataset_stats['folder_stats'].items() 
         if stats['missing_count'] > 0 and stats['label_count'] > 0],
        key=lambda x: x[1]/x[2], reverse=True
    )[:5]
    
    if problematic_folders:
        logger.warning("\n⚠️ 매칭률이 낮은 상위 5개 폴더:")
        for folder, missing, total in problematic_folders:
            rate = missing / total * 100
            logger.warning(f"  - {folder}: {missing}/{total} 누락 ({rate:.1f}%)")

    # 이미지가 없는 결과
    logger.info(f"\n✅ 총 라벨링된 이미지 수: {len(all_labeled_images)}")

    # ================== 데이터 분할 ==================
    random.seed(42)
    random.shuffle(all_labeled_images)

    n_total = len(all_labeled_images)
    n_train = int(config.SPLITS[0] * n_total)
    n_val = int(config.SPLITS[1] * n_total)
    n_test = n_total - n_train - n_val  # 잔여는 test로

    dataset_splits = {
        'train': all_labeled_images[:n_train],
        'val': all_labeled_images[n_train:n_train + n_val],
        'test': all_labeled_images[n_train + n_val:]
    }

    logger.info("\n📊 데이터 분할 정보")
    logger.info(f"  총 수       : {n_total}")
    logger.info(f"  Train       : {n_train}")
    logger.info(f"  Validation  : {n_val}")
    logger.info(f"  Test        : {n_test}")
    logger.info(f"  합계        : {n_train + n_val + n_test}")

    # ================== 이미지 복사 ==================
    for split_name, images in dataset_splits.items():
        logger.info(f"\n📦 {split_name} 세트: {len(images)}장 복사 중...")
        for item in tqdm(images):
            dst_path = os.path.join(config.PETA_OUTPUT_DIR, split_name, item["gender"], item["img_name"])
            shutil.copy2(item["src_path"], dst_path)

    # ================== 누락 보고 ==================
    logger.warning(f"\n⚠️ 누락된 라벨(이미지 없음): {len(missing_images)}")
    for folder, idx, gender in missing_images[:10]:
        logger.warning(f" - {folder} / {idx} ({gender})")
    if len(missing_images) > 10:
        logger.warning("... (이하 생략)")

    logger.info("\n✅ 전처리 완료! YOLOv8 학습 준비 완료.")
    
    return dataset_stats

if __name__ == "__main__":
    process_peta_dataset()
