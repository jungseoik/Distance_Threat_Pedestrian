from preprocess.preprocess_aihub import process_aihub_dataset
from preprocess.preprocess_peta import process_peta_dataset
import argparse
from utils.custom_logger import custom_logger

logger = custom_logger(__name__)

def main():
    """
    데이터셋 전처리 실행 스크립트
    
    사용법:
        python preprocess.py --dataset [aihub|peta|all]
    """
    parser = argparse.ArgumentParser(description='성별 분류 데이터셋 전처리 스크립트')
    parser.add_argument('-d' , '--dataset', type=str, choices=['aihub', 'peta', 'all'], default='all',
                        help='처리할 데이터셋 (aihub, peta, 또는 all)')
    
    args = parser.parse_args()
    
    logger.info("===== 성별 분류 데이터셋 전처리 시작 =====")
    
    if args.dataset in ['aihub', 'all']:
        logger.info("\n===== AI Hub 데이터셋 처리 시작 =====")
        process_aihub_dataset()
        logger.info("===== AI Hub 데이터셋 처리 완료 =====")
    
    if args.dataset in ['peta', 'all']:
        logger.info("\n===== PETA 데이터셋 처리 시작 =====")
        process_peta_dataset()
        logger.info("===== PETA 데이터셋 처리 완료 =====")
    
    logger.info("\n===== 모든 처리 완료 =====")

if __name__ == "__main__":
    main()