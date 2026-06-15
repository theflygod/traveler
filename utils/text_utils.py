"""
文本处理工具
提供文本清洗、格式化等功能
"""

import re
from typing import List, Dict, Any
from config.logging_config import get_logger

logger = get_logger('text_utils')


class TextUtils:
    """文本处理工具类"""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清洗文本：去除多余空白、标准化格式

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()

        return text

    @staticmethod
    def truncate_text(text: str, max_length: int = 200,
                      suffix: str = '...') -> str:
        """
        截断文本到指定长度

        Args:
            text: 原始文本
            max_length: 最大长度
            suffix: 截断后缀

        Returns:
            截断后的文本
        """
        if not text or len(text) <= max_length:
            return text

        return text[:max_length] + suffix

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """
        提取关键词（简单版）

        Args:
            text: 文本内容
            top_n: 返回关键词数量

        Returns:
            关键词列表
        """
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就',
            '不', '人', '都', '一', '一个', '上', '也', '很',
            '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '他', '她', '它', '们',
            '什么', '怎么', '如何', '哪里', '哪个', '多少',
            '可以', '这个', '那个', '这样', '那样', '因为', '所以'
        }

        words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)

        word_freq = {}
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        keywords = [word for word, freq in sorted_words[:top_n]]

        return keywords

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        计算文本相似度（简单Jaccard相似度）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度 (0-1)
        """
        words1 = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text1.lower()))
        words2 = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        similarity = len(intersection) / len(union)

        return similarity

    @staticmethod
    def highlight_keywords(text: str, keywords: List[str],
                           marker: str = '**') -> str:
        """
        高亮显示关键词

        Args:
            text: 原始文本
            keywords: 关键词列表
            marker: 高亮标记

        Returns:
            高亮后的文本
        """
        highlighted = text

        for keyword in keywords:
            if keyword and len(keyword) > 1:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                highlighted = pattern.sub(f'{marker}\\g<0>{marker}', highlighted)

        return highlighted

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        将文本分割为句子

        Args:
            text: 原始文本

        Returns:
            句子列表
        """
        sentences = re.split(r'[。！？!?；;]', text)

        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    @staticmethod
    def count_words(text: str) -> Dict[str, int]:
        """
        统计文本字数

        Args:
            text: 原始文本

        Returns:
            统计结果 {'chinese': 中文字数, 'english': 英文单词数, 'total': 总字符数}
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        total_chars = len(text)

        return {
            'chinese': chinese_chars,
            'english': english_words,
            'total': total_chars
        }

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        标准化空白字符

        Args:
            text: 原始文本

        Returns:
            标准化后的文本
        """
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()

        return text

    @staticmethod
    def remove_markdown_syntax(text: str) -> str:
        """
        移除Markdown语法标记

        Args:
            text: Markdown文本

        Returns:
            纯文本
        """
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        text = re.sub(r'`{1,3}(.+?)`{1,3}', r'\1', text)
        text = re.sub(r'^\s*[-*+]\s+', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

        return text.strip()
