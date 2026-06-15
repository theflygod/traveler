"""
系统配置管理
从环境变量加载配置信息
"""

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings:
    """系统配置类（单例模式）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # LLM 配置
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
        self.LLM_DEFAULT_MODEL = os.getenv('LLM_DEFAULT_MODEL', 'qwen-flash')
        self.LLM_DEFAULT_TEMPERATURE = float(os.getenv('LLM_DEFAULT_TEMPERATURE', '0.1'))

        # Embedding 配置
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-v4')
        self.EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1536'))

        # Milvus 配置
        self.MILVUS_URL = os.getenv('MILVUS_URL', 'http://localhost:19530')
        self.CHUNKS_COLLECTION = os.getenv('CHUNKS_COLLECTION', 'md_chunks')
        self.ENTITY_NAME_COLLECTION = os.getenv('ENTITY_NAME_COLLECTION', 'md_test')

        # 数据路径配置
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'files', '数据')

        # 验证必要配置
        self._validate()

        self._initialized = True

    def _validate(self):
        """验证必要配置是否存在"""
        required_configs = {
            'OPENAI_API_KEY': self.OPENAI_API_KEY,
            'OPENAI_API_BASE': self.OPENAI_API_BASE,
            'MILVUS_URL': self.MILVUS_URL,
        }

        missing = [key for key, value in required_configs.items() if not value]
        if missing:
            raise ValueError(f"缺少必要的环境变量配置: {', '.join(missing)}")

    def to_dict(self) -> dict:
        """将配置转换为字典"""
        return {
            'LLM': {
                'model': self.LLM_DEFAULT_MODEL,
                'temperature': self.LLM_DEFAULT_TEMPERATURE,
                'api_base': self.OPENAI_API_BASE,
            },
            'Embedding': {
                'model': self.EMBEDDING_MODEL,
                'dimension': self.EMBEDDING_DIM,
            },
            'Milvus': {
                'url': self.MILVUS_URL,
                'chunks_collection': self.CHUNKS_COLLECTION,
                'entity_collection': self.ENTITY_NAME_COLLECTION,
            },
            'Data': {
                'data_dir': self.DATA_DIR,
            }
        }

    def __repr__(self):
        return f"Settings(Milvus={self.MILVUS_URL}, LLM={self.LLM_DEFAULT_MODEL})"


# 全局配置实例
settings = Settings()
