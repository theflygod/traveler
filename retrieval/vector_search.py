"""
向量检索服务
基于Milvus进行语义相似度搜索
"""

from typing import List, Dict, Any, Optional
from data.embedding_service import EmbeddingService
from storage.milvus_client import MilvusClient
from config.logging_config import get_logger

logger = get_logger('vector_search')


class VectorSearch:
    """向量检索服务"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.milvus_client = MilvusClient()

    def search(self, query: str,
               top_k: int = 5,
               content_type_filter: str = None,
               city_filter: str = None) -> List[Dict[str, Any]]:
        """
        执行向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            content_type_filter: 内容类型过滤（景点介绍/线路推荐等）
            city_filter: 城市过滤

        Returns:
            检索结果列表
        """
        try:
            query_vector = self.embedding_service.embed_text(query)

            keyword_filter = {}
            if content_type_filter:
                keyword_filter['content_type'] = content_type_filter
            if city_filter:
                keyword_filter['city'] = city_filter

            results = self.milvus_client.hybrid_search(
                query_vector=query_vector,
                keyword_filter=keyword_filter if keyword_filter else None,
                top_k=top_k
            )

            logger.info(f"向量检索完成: query='{query}', 返回{len(results)}条结果")
            return results

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise

    def search_by_intent(self, query: str, intent: str,
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        根据意图类型执行检索

        Args:
            query: 查询文本
            intent: 意图类型（景点介绍/线路检索/实用信息/知识问答）
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        content_type_map = {
            '景点介绍': '景点介绍',
            '线路检索': '线路推荐',
            '实用信息': None,
            '知识问答': None
        }

        content_type = content_type_map.get(intent)

        return self.search(
            query=query,
            top_k=top_k,
            content_type_filter=content_type
        )

    def search_attractions(self, query: str, city: str = None,
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索景点介绍

        Args:
            query: 查询文本
            city: 城市名称
            top_k: 返回结果数量

        Returns:
            景点检索结果
        """
        return self.search(
            query=query,
            top_k=top_k,
            content_type_filter='景点介绍',
            city_filter=city
        )

    def search_routes(self, query: str, city: str = None,
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索线路推荐

        Args:
            query: 查询文本
            city: 城市名称
            top_k: 返回结果数量

        Returns:
            线路检索结果
        """
        return self.search(
            query=query,
            top_k=top_k,
            content_type_filter='线路推荐',
            city_filter=city
        )

    def search_practical_info(self, query: str, city: str = None,
                              top_k: int = 8) -> List[Dict[str, Any]]:
        """
        搜索实用信息（酒店、美食、交通等）

        Args:
            query: 查询文本
            city: 城市名称
            top_k: 返回结果数量

        Returns:
            实用信息检索结果
        """
        return self.search(
            query=query,
            top_k=top_k,
            content_type_filter=None,
            city_filter=city
        )
