import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
def custom_logger(name: str) -> logging.Logger:
    """
    커스텀 로거를 생성합니다.
    콘솔 핸들러와 파일 핸들러가 동시에 작동하며, 각각 다른 로그 레벨을 가집니다.

    Args:
        name (str): 로거 이름 (보통 __name__ 사용)

    Returns:
        logging.Logger: 설정된 Logger 객체

    로그 레벨 설정:
        - 콘솔 핸들러: INFO 레벨 (INFO, WARNING, ERROR, CRITICAL만 출력)
        - 파일 핸들러: DEBUG 레벨 (모든 레벨 기록)
        - 로그 파일은 'logs' 디렉토리에 로거 이름별로 저장되며 5일치만 보관

    사용 예시:
        ```python
        logger = custom_logger(__name__)
        
        logger.debug("디버그 메시지")    # 파일에만 기록
        logger.info("정보 메시지")       # 콘솔 출력 + 파일 기록
        logger.warning("경고 메시지")    # 콘솔 출력 + 파일 기록
        logger.error("에러 메시지")      # 콘솔 출력 + 파일 기록
        ```

    출력 형식:
        [YYYY-MM-DD HH:MM:SS] [레벨] [모듈:라인] [함수명] 메시지
    """
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 이미 핸들러가 있다면 추가하지 않음
    if logger.handlers:
        return logger

    # logs 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_format = "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] [%(funcName)s] %(message)s"
    datafmt = "%Y-%m-%d %H:%M:%S"
    # 포맷터 설정
    console_formatter = logging.Formatter(
        log_format,
        datefmt=datafmt,
    )
    file_formatter = logging.Formatter(
        log_format,
        datefmt=datafmt,
    )
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # 파일 핸들러
    file_handler = TimedRotatingFileHandler(
        filename=log_dir / f"{name}.log",
        when="midnight",
        interval=1,
        backupCount=5,  # 5일치만 보관
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
