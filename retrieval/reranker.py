"""
结果重排序服务
对检索结果进行二次排序，提升相关性
"""

from typing import List, Dict, Any
from config.logging_config import get_logger

logger = get_logger('reranker')


class Reranker:
    """结果重排序器"""

    def __init__(self):
        pass

    def rerank(self, query: str,
               results: List[Dict[str, Any]],
               strategy: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            results: 原始检索结果
            strategy: 重排序策略（hybrid/title_boost/content_match）

        Returns:
            重排序后的结果
        """
        if not results:
            return []

        if strategy == 'title_boost':
            return self._rerank_by_title_boost(query, results)
        elif strategy == 'content_match':
            return self._rerank_by_content_match(query, results)
        else:
            return self._rerank_hybrid(query, results)

    def _rerank_by_title_boost(self, query: str,
                               results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于标题加权的重排序

        Args:
            query: 查询文本
            results: 原始结果

        Returns:
            重排序结果
        """
        query_lower = query.lower()

        for result in results:
            title = result.get('section_title', '').lower()
            attraction = result.get('attraction_name', '').lower()
            original_score = result.get('score', 0) or result.get('final_score', 0)

            boost_factor = 1.0

            if query_lower in title:
                boost_factor = 2.0
            elif any(word in title for word in query_lower.split() if len(word) > 1):
                boost_factor = 1.5

            if query_lower in attraction:
                boost_factor *= 1.8

            result['rerank_score'] = original_score * boost_factor
            result['boost_factor'] = boost_factor

        results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.info(f"标题加权重排序完成，返回{len(results)}条结果")
        return results

    def _rerank_by_content_match(self, query: str,
                                 results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于内容匹配度的重排序

        Args:
            query: 查询文本
            results: 原始结果

        Returns:
            重排序结果
        """
        query_words = [w for w in query.lower().split() if len(w) > 1]

        for result in results:
            content = result.get('content', '').lower()
            original_score = result.get('score', 0) or result.get('final_score', 0)

            match_count = sum(1 for word in query_words if word in content)
            match_ratio = match_count / len(query_words) if query_words else 0

            match_score = original_score * (1 + match_ratio)

            result['rerank_score'] = match_score
            result['match_ratio'] = match_ratio

        results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.info(f"内容匹配重排序完成，返回{len(results)}条结果")
        return results

    def _rerank_hybrid(self, query: str,
                       results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        混合重排序策略

        Args:
            query: 查询文本
            results: 原始结果

        Returns:
            重排序结果
        """
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 1]

        for result in results:
            title = result.get('section_title', '').lower()
            content = result.get('content', '').lower()
            attraction = result.get('attraction_name', '').lower()
            original_score = result.get('score', 0) or result.get('final_score', 0)

            title_score = 0
            if query_lower in title:
                title_score = 3.0
            else:
                title_score = sum(1 for word in query_words if word in title) * 1.0

            attraction_score = 2.0 if query_lower in attraction else 0

            content_match_ratio = sum(1 for word in query_words if word in content) / len(
                query_words) if query_words else 0
            content_score = content_match_ratio * 1.5

            final_score = original_score + title_score + attraction_score + content_score

            result['rerank_score'] = final_score
            result['title_score'] = title_score
            result['attraction_score'] = attraction_score
            result['content_score'] = content_score

        results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.info(f"混合重排序完成，返回{len(results)}条结果")
        return results

    def filter_by_relevance(self, results: List[Dict[str, Any]],
                            min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        根据相关性分数过滤结果

        Args:
            results: 检索结果
            min_score: 最低分数阈值

        Returns:
            过滤后的结果
        """
        filtered = [r for r in results if r.get('rerank_score', r.get('score', 0)) >= min_score]

        logger.info(f"相关性过滤: {len(results)} -> {len(filtered)} (min_score={min_score})")
        return filtered

    def deduplicate_results(self, results: List[Dict[str, Any]],
                            threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        去重相似结果

        Args:
            results: 检索结果
            threshold: 相似度阈值

        Returns:
            去重后的结果
        """
        seen_contents = set()
        unique_results = []

        for result in results:
            content = result.get('content', '')[:200]
            content_hash = hash(content)

            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)

        logger.info(f"去重: {len(results)} -> {len(unique_results)}")
        return unique_results
