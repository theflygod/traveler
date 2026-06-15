"""
混合检索服务
结合向量检索和关键词匹配，提升检索准确率
"""

import re
from typing import List, Dict, Any, Tuple
from data.embedding_service import EmbeddingService
from storage.milvus_client import MilvusClient
from config.logging_config import get_logger

logger = get_logger('hybrid_search')


class HybridSearch:
    """混合检索服务"""

    def __init__(self, vector_weight: float = 0.7, keyword_weight: float = 0.3):
        """
        初始化混合检索

        Args:
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.embedding_service = EmbeddingService()
        self.milvus_client = MilvusClient()

    def search(self, query: str,
               top_k: int = 5,
               filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        执行混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            混合检索结果列表
        """
        try:
            vector_results = self._vector_search(query, top_k * 2, filters)
            keyword_results = self._keyword_search(query, top_k * 2, filters)

            combined_results = self._combine_results(
                vector_results,
                keyword_results,
                top_k
            )

            logger.info(f"混合检索完成: query='{query}', 返回{len(combined_results)}条结果")
            return combined_results

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise

    def _vector_search(self, query: str, top_k: int,
                       filters: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        执行向量检索部分

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            向量检索结果
        """
        query_vector = self.embedding_service.embed_text(query)

        results = self.milvus_client.hybrid_search(
            query_vector=query_vector,
            keyword_filter=filters,
            top_k=top_k
        )

        for result in results:
            result['vector_score'] = result.get('score', 0)

        return results

    def _keyword_search(self, query: str, top_k: int,
                        filters: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        执行关键词检索部分

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            关键词检索结果
        """
        keywords = self._extract_keywords(query)

        if not keywords:
            return []

        filter_expr = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value and value.strip():
                    conditions.append(f'{key} == "{value}"')
            if conditions:
                filter_expr = " and ".join(conditions)

        output_fields = [
            "chunk_id", "content", "content_type", "city",
            "attraction_name", "route_name", "section_title",
            "source_filename"
        ]

        try:
            results = self.milvus_client.search_by_vector(
                query_vector=[0.0] * self.embedding_service.get_dimension(),
                top_k=top_k,
                filter_expr=filter_expr,
                output_fields=output_fields
            )

            keyword_scored_results = []
            for result in results:
                content = result.get('content', '')
                title = result.get('section_title', '')
                text = f"{title} {content}".lower()

                score = self._calculate_keyword_score(keywords, text)

                if score > 0:
                    result['keyword_score'] = score
                    keyword_scored_results.append(result)

            keyword_scored_results.sort(
                key=lambda x: x['keyword_score'],
                reverse=True
            )

            return keyword_scored_results[:top_k]

        except Exception as e:
            logger.warning(f"关键词检索失败: {e}")
            return []

    def _extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词

        Args:
            query: 查询文本

        Returns:
            关键词列表
        """
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就',
                     '不', '人', '都', '一', '一个', '上', '也', '很',
                     '到', '说', '要', '去', '你', '会', '着', '没有',
                     '看', '好', '自己', '这', '他', '她', '它', '们',
                     '什么', '怎么', '如何', '哪里', '哪个', '多少'}

        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query)

        keywords = [word for word in words if word not in stopwords and len(word) > 1]

        return keywords

    def _calculate_keyword_score(self, keywords: List[str],
                                 text: str) -> float:
        """
        计算关键词匹配分数

        Args:
            keywords: 关键词列表
            text: 待评分文本

        Returns:
            匹配分数 (0-1)
        """
        if not keywords or not text:
            return 0.0

        match_count = sum(1 for keyword in keywords if keyword in text)

        score = match_count / len(keywords)

        return score

    def _combine_results(self,
                         vector_results: List[Dict[str, Any]],
                         keyword_results: List[Dict[str, Any]],
                         top_k: int) -> List[Dict[str, Any]]:
        """
        合并向量检索和关键词检索结果

        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            top_k: 最终返回数量

        Returns:
            合并后的结果
        """
        all_results = {}

        for result in vector_results:
            chunk_id = result.get('chunk_id', '')
            vector_score = result.get('vector_score', 0)

            all_results[chunk_id] = {
                **result,
                'final_score': vector_score * self.vector_weight,
                'vector_score': vector_score,
                'keyword_score': 0
            }

        for result in keyword_results:
            chunk_id = result.get('chunk_id', '')
            keyword_score = result.get('keyword_score', 0)

            if chunk_id in all_results:
                all_results[chunk_id]['keyword_score'] = keyword_score
                all_results[chunk_id]['final_score'] += keyword_score * self.keyword_weight
            else:
                all_results[chunk_id] = {
                    **result,
                    'final_score': keyword_score * self.keyword_weight,
                    'vector_score': 0,
                    'keyword_score': keyword_score
                }

        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )

        final_results = []
        for result in sorted_results[:top_k]:
            final_result = {k: v for k, v in result.items()
                            if k not in ['vector_score', 'keyword_score']}
            final_result['final_score'] = result['final_score']
            final_results.append(final_result)

        return final_results

    def search_with_rerank(self, query: str,
                           top_k: int = 5,
                           filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        带重排序的混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            重排序后的结果
        """
        results = self.search(query, top_k * 2, filters)

        reranked_results = self._rerank_by_relevance(query, results)

        return reranked_results[:top_k]

    def _rerank_by_relevance(self, query: str,
                             results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据相关性重排序

        Args:
            query: 查询文本
            results: 原始结果

        Returns:
            重排序后的结果
        """
        query_lower = query.lower()

        scored_results = []
        for result in results:
            title = result.get('section_title', '').lower()
            content = result.get('content', '').lower()
            attraction = result.get('attraction_name', '').lower()

            relevance_score = 0

            if query_lower in title:
                relevance_score += 3

            if query_lower in attraction:
                relevance_score += 2

            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 1 and word in content:
                    relevance_score += 1

            result['relevance_score'] = relevance_score
            scored_results.append(result)

        scored_results.sort(
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )

        return scored_results
