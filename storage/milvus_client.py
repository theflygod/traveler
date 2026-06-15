"""
Milvus数据库客户端
提供向量数据库的连接、插入、查询等操作
"""

import uuid
import json
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, utility
from config.settings import settings
from config.logging_config import get_logger
from .schema import MilvusSchema

logger = get_logger('milvus_client')


class MilvusClient:
    """Milvus数据库客户端（单例模式）"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.collection_name = settings.CHUNKS_COLLECTION
        self.embedding_dim = settings.EMBEDDING_DIM

        try:
            connections.connect(
                alias="default",
                uri=settings.MILVUS_URL
            )
            logger.info(f"成功连接到Milvus: {settings.MILVUS_URL}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

        self.collection = self._get_or_create_collection()
        self._initialized = True

    def _get_or_create_collection(self) -> Collection:
        """
        获取或创建集合

        Returns:
            Collection对象
        """
        if utility.has_collection(self.collection_name):
            logger.info(f"集合已存在: {self.collection_name}")
            collection = Collection(self.collection_name)

            try:
                collection.load()
            except Exception as e:
                logger.warning(f"集合加载失败，尝试重建索引: {e}")
                self._rebuild_index(collection)
                collection.load()

            return collection

        logger.info(f"创建新集合: {self.collection_name}")
        schema = MilvusSchema.get_chunks_schema()
        collection = Collection(name=self.collection_name, schema=schema)

        self._create_index_for_collection(collection)

        collection.load()
        logger.info(f"集合并创建索引完成: {self.collection_name}")

        return collection

    def _create_index_for_collection(self, collection: Collection):
        """
        为集合创建索引（兼容新版pymilvus）

        Args:
            collection: Collection对象
        """
        try:
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            }

            collection.create_index(
                field_name="vector",
                index_params=index_params
            )

            logger.info("索引创建成功")
        except Exception as e:
            logger.error(f"索引创建失败: {e}")
            raise

    def _rebuild_index(self, collection: Collection):
        """
        重建索引

        Args:
            collection: Collection对象
        """
        try:
            indexes = collection.indexes
            for index in indexes:
                index.drop()
                logger.info(f"删除旧索引")

            self._create_index_for_collection(collection)
            logger.info(f"索引重建完成")
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            raise

    # ... existing code ...

    def insert_chunks(self, chunks_data: List[Dict[str, Any]],
                      vectors: List[List[float]]) -> List[int]:
        """
        批量插入文档分块

        Args:
            chunks_data: 分块数据列表（字典格式）
            vectors: 对应的向量列表

        Returns:
            插入的主键ID列表
        """
        if not chunks_data or not vectors:
            logger.warning("没有数据需要插入")
            return []

        if len(chunks_data) != len(vectors):
            raise ValueError(f"数据数量不匹配: chunks={len(chunks_data)}, vectors={len(vectors)}")

        try:
            entities = []
            for chunk_data, vector in zip(chunks_data, vectors):
                entity = {
                    "chunk_id": chunk_data.get('chunk_id', str(uuid.uuid4())),
                    "content": chunk_data.get('content', ''),
                    "content_type": chunk_data.get('content_type', ''),
                    "city": chunk_data.get('city', ''),
                    "attraction_name": chunk_data.get('attraction_name', ''),
                    "route_name": chunk_data.get('route_name', ''),
                    "hotel_name": chunk_data.get('hotel_name', ''),
                    "restaurant_name": chunk_data.get('restaurant_name', ''),
                    "section_title": chunk_data.get('section_title', ''),
                    "source_filename": chunk_data.get('source_filename', ''),
                    "source_path": chunk_data.get('source_path', ''),
                    "vector": vector
                }
                entities.append(entity)

            insert_result = self.collection.insert(entities)
            ids = insert_result.primary_keys

            logger.info(f"成功插入 {len(ids)} 条记录")
            return ids

        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            raise

    def search_by_vector(self, query_vector: List[float],
                         top_k: int = 5,
                         filter_expr: str = None,
                         output_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        通过向量搜索相似文档

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filter_expr: 过滤表达式（如 "city == '三亚'"）
            output_fields: 需要返回的字段列表

        Returns:
            搜索结果列表
        """
        if output_fields is None:
            output_fields = [
                "chunk_id", "content", "content_type", "city",
                "attraction_name", "route_name", "section_title",
                "source_filename"
            ]

        try:
            search_params = MilvusSchema.get_search_params()

            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )

            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        'score': hit.score,
                        'distance': hit.distance,
                    }
                    for field in output_fields:
                        result[field] = hit.entity.get(field, '')
                    formatted_results.append(result)

            logger.info(f"向量搜索返回 {len(formatted_results)} 条结果")
            return formatted_results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            raise

    def hybrid_search(self, query_vector: List[float],
                      keyword_filter: Dict[str, str] = None,
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        混合检索：结合向量搜索和元数据过滤

        Args:
            query_vector: 查询向量
            keyword_filter: 关键词过滤条件（如 {"city": "三亚", "content_type": "景点介绍"}）
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        filter_expr = None
        if keyword_filter:
            conditions = []
            for key, value in keyword_filter.items():
                if value and value.strip():
                    conditions.append(f'{key} == "{value}"')
            if conditions:
                filter_expr = " and ".join(conditions)

        return self.search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            filter_expr=filter_expr
        )

    def delete_by_filter(self, filter_expr: str) -> int:
        """
        根据过滤条件删除数据

        Args:
            filter_expr: 过滤表达式

        Returns:
            删除的记录数
        """
        try:
            result = self.collection.delete(filter_expr)
            logger.info(f"删除记录: {filter_expr}")
            return result.delete_count
        except Exception as e:
            logger.error(f"删除数据失败: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            统计信息字典
        """
        try:
            self.collection.flush()
            stats = {
                "collection_name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "is_loaded": self.collection.is_empty
            }
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            raise

    def drop_collection(self):
        """删除集合（谨慎使用）"""
        try:
            self.collection.drop()
            logger.warning(f"集合已删除: {self.collection_name}")
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            raise

    def close(self):
        """关闭连接"""
        try:
            connections.disconnect("default")
            logger.info("Milvus连接已关闭")
        except Exception as e:
            logger.error(f"关闭连接失败: {e}")
