"""
Markdown文档解析器
负责读取和解析Markdown文件夹，按二级标题分块
"""

import os
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """文档分块数据类"""
    content: str  # 内容正文
    content_type: str = ""  # 内容类型
    city: str = ""  # 城市/地区
    attraction_name: str = ""  # 景点名称
    route_name: str = ""  # 线路名称
    hotel_name: str = ""  # 酒店名称
    restaurant_name: str = ""  # 餐厅名称
    source_filename: str = ""  # 来源文件名
    source_path: str = ""  # 来源路径
    section_title: str = ""  # 章节标题
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.content,
            'content_type': self.content_type,
            'city': self.city,
            'attraction_name': self.attraction_name,
            'route_name': self.route_name,
            'hotel_name': self.hotel_name,
            'restaurant_name': self.restaurant_name,
            'source_filename': self.source_filename,
            'source_path': self.source_path,
            'section_title': self.section_title,
            **self.metadata
        }


class DocumentParser:
    """Markdown文档解析器"""

    def __init__(self):
        self.supported_extensions = ['.md']

    def parse_file(self, file_path: str) -> List[DocumentChunk]:
        """
        解析单个Markdown文件

        Args:
            file_path: Markdown文件路径

        Returns:
            文档分块列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if not file_path.endswith('.md'):
            raise ValueError(f"不支持的文件格式: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(file_path)
        chunks = self._split_by_h2(content, file_path, filename)

        return chunks

    def parse_directory(self, dir_path: str) -> List[DocumentChunk]:
        """
        递归解析目录下所有Markdown文件

        Args:
            dir_path: 目录路径

        Returns:
            所有文档的分块列表
        """
        all_chunks = []

        for root, dirs, files in os.walk(dir_path):
            for filename in files:
                if filename.endswith('.md'):
                    file_path = os.path.join(root, filename)
                    try:
                        chunks = self.parse_file(file_path)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        print(f"解析文件失败 {file_path}: {e}")

        return all_chunks

    def _split_by_h2(self, content: str, source_path: str,
                     source_filename: str) -> List[DocumentChunk]:
        """
        按二级标题分割文档

        Args:
            content: 文档全文
            source_path: 文件路径
            source_filename: 文件名

        Returns:
            分块列表
        """
        lines = content.split('\n')
        chunks = []

        current_section = None
        current_content = []

        for line in lines:
            if line.startswith('## '):
                if current_section is not None:
                    chunk_text = '\n'.join(current_content).strip()
                    if chunk_text:
                        chunk = self._create_chunk(
                            content=chunk_text,
                            section_title=current_section,
                            source_path=source_path,
                            source_filename=source_filename
                        )
                        chunks.append(chunk)

                current_section = line[3:].strip()
                current_content = [line]
            else:
                current_content.append(line)

        if current_section and current_content:
            chunk_text = '\n'.join(current_content).strip()
            if chunk_text:
                chunk = self._create_chunk(
                    content=chunk_text,
                    section_title=current_section,
                    source_path=source_path,
                    source_filename=source_filename
                )
                chunks.append(chunk)

        return chunks

    def _create_chunk(self, content: str, section_title: str,
                      source_path: str, source_filename: str) -> DocumentChunk:
        """
        创建文档分块对象

        Args:
            content: 分块内容
            section_title: 章节标题
            source_path: 文件路径
            source_filename: 文件名

        Returns:
            文档分块对象
        """
        chunk = DocumentChunk(
            content=content,
            section_title=section_title,
            source_filename=source_filename,
            source_path=source_path
        )

        return chunk
