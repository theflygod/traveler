"""
回答生成模块
基于检索结果生成自然语言回答
"""

from .llm_client import LLMClient
from .answer_generator import AnswerGenerator
from .prompts import ANSWER_GENERATION_PROMPT

__all__ = ['LLMClient', 'AnswerGenerator', 'ANSWER_GENERATION_PROMPT']
