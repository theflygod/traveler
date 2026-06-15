"""
存储模块
包含Milvus数据库客户端和表结构定义
"""

from .schema import MilvusSchema
from .milvus_client import MilvusClient

__all__ = ['MilvusSchema', 'MilvusClient']
