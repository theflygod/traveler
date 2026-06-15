"""
元数据提取器
从文档分块中提取结构化元数据
"""

import re
from typing import Dict, Any, Optional
from .document_parser import DocumentChunk


class MetadataExtractor:
    """元数据提取器"""

    def __init__(self):
        self.content_type_keywords = {
            '景点介绍': ['景点', '景区', '目的地', '观光'],
            '线路推荐': ['线路', '行程', '路线', '游玩', '安排'],
            '酒店信息': ['酒店', '住宿', '民宿', '客栈'],
            '美食推荐': ['美食', '餐厅', '小吃', '餐饮', '特色菜'],
            '交通指南': ['交通', '机场', '高铁', '火车', '公交', '打车'],
            '文化民俗': ['文化', '民俗', '历史', '传统']
        }

    def extract_metadata(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        从文档分块中提取元数据

        Args:
            chunk: 文档分块对象

        Returns:
            填充了元数据的分块对象
        """
        chunk = self._extract_from_metadata_section(chunk)
        chunk = self._extract_from_filename(chunk)
        chunk = self._extract_from_content(chunk)

        return chunk

    def _extract_from_metadata_section(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        从文档的元数据章节提取信息

        Args:
            chunk: 文档分块对象

        Returns:
            更新后的分块对象
        """
        content = chunk.content

        metadata_patterns = {
            'content_type': r'内容类型[：:]\s*(.+)',
            'city': r'城市[：:]\s*(.+)',
            'attraction_name': r'景点名称[：:]\s*(.+)',
            'route_name': r'线路名称[：:]\s*(.+)',
            'hotel_name': r'酒店名称[：:]\s*(.+)',
            'restaurant_name': r'餐厅名称[：:]\s*(.+)',
        }

        for key, pattern in metadata_patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1).strip()
                setattr(chunk, key, value)

        return chunk

    def _extract_from_filename(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        从文件名推断元数据

        Args:
            chunk: 文档分块对象

        Returns:
            更新后的分块对象
        """
        filename = chunk.source_filename

        city_pattern = r'(三亚|云南|厦门|张家界|成都|杭州)'
        content_type_pattern = r'(景点推荐|线路推荐|美食推荐|住宿推荐|交通指南)'

        city_match = re.search(city_pattern, filename)
        if city_match and not chunk.city:
            chunk.city = city_match.group(1)

        type_match = re.search(content_type_pattern, filename)
        if type_match and not chunk.content_type:
            type_mapping = {
                '景点推荐': '景点介绍',
                '线路推荐': '线路推荐',
                '美食推荐': '美食推荐',
                '住宿推荐': '酒店信息',
                '交通指南': '交通指南'
            }
            chunk.content_type = type_mapping.get(type_match.group(1), '')

        return chunk

    def _extract_from_content(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        从内容中提取元数据

        Args:
            chunk: 文档分块对象

        Returns:
            更新后的分块对象
        """
        content = chunk.content

        if not chunk.city:
            city_pattern = r'(三亚|云南|厦门|张家界|成都|杭州)'
            match = re.search(city_pattern, content)
            if match:
                chunk.city = match.group(1)

        if not chunk.content_type:
            chunk.content_type = self._infer_content_type(content)

        return chunk

    def _infer_content_type(self, content: str) -> str:
        """
        根据内容推断内容类型

        Args:
            content: 文本内容

        Returns:
            内容类型
        """
        scores = {}

        for content_type, keywords in self.content_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                scores[content_type] = score

        if scores:
            return max(scores, key=scores.get)

        return '其他'
