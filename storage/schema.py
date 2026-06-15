"""
Milvus数据库表结构定义
定义向量集合的字段、索引和参数
"""

from pymilvus import DataType, FieldSchema, CollectionSchema
from config.settings import settings


class MilvusSchema:
    """Milvus数据库Schema定义"""

    @staticmethod
    def get_chunks_schema():
        """
        获取文档分块集合的Schema

        Returns:
            CollectionSchema对象
        """
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="主键ID（自动生成）"
            ),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="分块唯一标识"
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="内容正文"
            ),
            FieldSchema(
                name="content_type",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="内容类型（景点介绍/线路推荐/酒店信息/美食推荐/交通指南）"
            ),
            FieldSchema(
                name="city",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="城市/地区"
            ),
            FieldSchema(
                name="attraction_name",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="景点名称"
            ),
            FieldSchema(
                name="route_name",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="线路名称"
            ),
            FieldSchema(
                name="hotel_name",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="酒店名称"
            ),
            FieldSchema(
                name="restaurant_name",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="餐厅名称"
            ),
            FieldSchema(
                name="section_title",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="章节标题"
            ),
            FieldSchema(
                name="source_filename",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="来源文件名"
            ),
            FieldSchema(
                name="source_path",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="来源路径"
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.EMBEDDING_DIM,
                description="文本向量表示"
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="旅游知识库文档分块集合",
            enable_dynamic_field=True
        )

        return schema

    @staticmethod
    def get_index_params():
        """
        获取向量索引参数（兼容新版pymilvus）

        Returns:
            索引参数字典
        """
        return {
            "field_name": "vector",
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }

    @staticmethod
    def get_search_params(metric_type: str = "COSINE"):
        """
        获取搜索参数

        Args:
            metric_type: 相似度度量方式（COSINE/IP/L2）

        Returns:
            搜索参数字典
        """
        return {
            "metric_type": metric_type,
            "params": {"nprobe": 10}
        }
