"""
日志配置
统一日志格式和输出
"""

import logging
import os
from datetime import datetime


def setup_logging(
        log_level: str = 'INFO',
        log_dir: str = None,
        log_file: str = None
) -> logging.Logger:
    """
    配置日志系统

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志目录
        log_file: 日志文件名

    Returns:
        配置好的logger实例
    """
    if log_dir is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        log_dir = os.path.join(project_root, 'logs')

    os.makedirs(log_dir, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = f'travel_kb_{timestamp}.log'

    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger('travel_kb')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    获取logger实例

    Args:
        name: logger名称

    Returns:
        logger实例
    """
    if name:
        full_name = f'travel_kb.{name}'
    else:
        full_name = 'travel_kb'

    return logging.getLogger(full_name)
