"""
文件操作工具
提供文件读写、目录遍历等功能
"""

import os
import json
from typing import List, Dict, Any
from config.logging_config import get_logger

logger = get_logger('file_utils')


class FileUtils:
    """文件操作工具类"""

    @staticmethod
    def read_file(file_path: str, encoding: str = 'utf-8') -> str:
        """
        读取文件内容

        Args:
            file_path: 文件路径
            encoding: 编码格式

        Returns:
            文件内容
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            logger.debug(f"读取文件成功: {file_path}")
            return content
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            raise

    @staticmethod
    def write_file(file_path: str, content: str, encoding: str = 'utf-8'):
        """
        写入文件内容

        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 编码格式
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            logger.debug(f"写入文件成功: {file_path}")
        except Exception as e:
            logger.error(f"写入文件失败 {file_path}: {e}")
            raise

    @staticmethod
    def read_json(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        读取JSON文件

        Args:
            file_path: 文件路径
            encoding: 编码格式

        Returns:
            JSON数据
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            logger.debug(f"读取JSON成功: {file_path}")
            return data
        except Exception as e:
            logger.error(f"读取JSON失败 {file_path}: {e}")
            raise

    @staticmethod
    def write_json(file_path: str, data: Dict[str, Any],
                   indent: int = 2, encoding: str = 'utf-8'):
        """
        写入JSON文件

        Args:
            file_path: 文件路径
            data: JSON数据
            indent: 缩进空格数
            encoding: 编码格式
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding=encoding) as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            logger.debug(f"写入JSON成功: {file_path}")
        except Exception as e:
            logger.error(f"写入JSON失败 {file_path}: {e}")
            raise

    @staticmethod
    def list_files(directory: str, extension: str = None) -> List[str]:
        """
        列出目录下的文件

        Args:
            directory: 目录路径
            extension: 文件扩展名过滤（如 '.md'）

        Returns:
            文件路径列表
        """
        try:
            files = []
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    if extension and not filename.endswith(extension):
                        continue
                    file_path = os.path.join(root, filename)
                    files.append(file_path)

            logger.debug(f"列出文件: {directory}, 共{len(files)}个")
            return files
        except Exception as e:
            logger.error(f"列出文件失败 {directory}: {e}")
            raise

    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        获取文件大小

        Args:
            file_path: 文件路径

        Returns:
            文件大小（字节）
        """
        try:
            size = os.path.getsize(file_path)
            return size
        except Exception as e:
            logger.error(f"获取文件大小失败 {file_path}: {e}")
            raise

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        格式化文件大小

        Args:
            size_bytes: 文件大小（字节）

        Returns:
            格式化后的字符串
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    @staticmethod
    def ensure_directory(directory: str):
        """
        确保目录存在，不存在则创建

        Args:
            directory: 目录路径
        """
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"目录已确保存在: {directory}")
        except Exception as e:
            logger.error(f"创建目录失败 {directory}: {e}")
            raise

    @staticmethod
    def delete_file(file_path: str) -> bool:
        """
        删除文件

        Args:
            file_path: 文件路径

        Returns:
            是否删除成功
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"删除文件成功: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除文件失败 {file_path}: {e}")
            raise
