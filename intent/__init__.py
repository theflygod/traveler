"""
意图识别模块
识别用户查询的意图类型
"""

from .classifier import IntentClassifier
from .prompts import INTENT_CLASSIFICATION_PROMPT

__all__ = ['IntentClassifier', 'INTENT_CLASSIFICATION_PROMPT']
