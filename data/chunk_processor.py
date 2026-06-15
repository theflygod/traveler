"""
文档分块处理器
负责对文档分块进行清洗、优化和后处理
"""

import re
from typing import List
from .document_parser import DocumentChunk
from .metadata_extractor import MetadataExtractor


class ChunkProcessor:
    """文档分块处理器"""

    def __init__(self):
        self.metadata_extractor = MetadataExtractor()

    def process_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        批量处理文档分块

        Args:
            chunks: 原始分块列表

        Returns:
            处理后的分块列表
        """
        processed = []

        for chunk in chunks:
            try:
                processed_chunk = self.process_single_chunk(chunk)
                processed.append(processed_chunk)
            except Exception as e:
                print(f"处理分块失败: {e}, 分块标题: {chunk.section_title}")

        return processed

    def process_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        处理单个文档分块

        Args:
            chunk: 原始分块

        Returns:
            处理后的分块
        """
        chunk = self._clean_content(chunk)
        chunk = self.metadata_extractor.extract_metadata(chunk)
        chunk = self._optimize_content(chunk)

        return chunk

    def _clean_content(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        清洗内容：去除多余空白、标准化格式

        Args:
            chunk: 文档分块

        Returns:
            清洗后的分块
        """
        content = chunk.content

        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        content = content.strip()

        chunk.content = content
        return chunk

    def _optimize_content(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        优化内容：确保关键信息完整

        Args:
            chunk: 文档分块

        Returns:
            优化后的分块
        """
        if chunk.city and chunk.city not in chunk.content[:200]:
            chunk.content = f"【{chunk.city}】\n\n{chunk.content}"

        if chunk.content_type and chunk.content_type not in chunk.content[:200]:
            chunk.metadata['processed_type'] = chunk.content_type

        return chunk
