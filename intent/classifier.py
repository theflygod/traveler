"""
意图分类器
使用LLM识别用户查询的意图类型
"""

import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from config.settings import settings
from config.logging_config import get_logger
from .prompts import INTENT_CLASSIFICATION_PROMPT, INTENT_EXAMPLES

logger = get_logger('intent_classifier')


class IntentClassifier:
    """意图分类器"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        self.model = settings.LLM_DEFAULT_MODEL
        self.temperature = 0.1

    def classify(self, query: str) -> Dict[str, Any]:
        """
        识别用户查询的意图

        Args:
            query: 用户查询文本

        Returns:
            意图识别结果字典
        """
        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个旅游智能助手的意图识别专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=200
            )

            result_text = response.choices[0].message.content.strip()

            intent_result = self._parse_response(result_text)

            if intent_result:
                logger.info(f"意图识别成功: query='{query}', intent={intent_result.get('intent')}")
                return intent_result
            else:
                logger.warning(f"意图解析失败，使用默认规则: query='{query}'")
                return self._rule_based_classify(query)

        except Exception as e:
            logger.error(f"意图识别失败: {e}，使用规则分类")
            return self._rule_based_classify(query)

    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        解析LLM响应

        Args:
            response_text: LLM返回的文本

        Returns:
            解析后的意图结果
        """
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                required_fields = ['intent', 'confidence']
                if all(field in result for field in required_fields):
                    return result

            return None

        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return None

    def _rule_based_classify(self, query: str) -> Dict[str, Any]:
        """
        基于规则的意图分类（降级方案）

        Args:
            query: 用户查询文本

        Returns:
            意图识别结果
        """
        query_lower = query.lower()

        attraction_keywords = ['景点', '好玩', '值得去', '推荐', '必去', '景区', '地方']
        route_keywords = ['线路', '行程', '几天', '日游', '安排', '路线', '怎么玩', '规划']
        practical_keywords = ['门票', '价格', '多少钱', '开放时间', '交通', '住宿', '酒店', '美食', '好吃']

        scores = {
            '景点介绍': sum(1 for kw in attraction_keywords if kw in query_lower),
            '线路检索': sum(1 for kw in route_keywords if kw in query_lower),
            '实用信息': sum(1 for kw in practical_keywords if kw in query_lower),
            '知识问答': 1
        }

        intent = max(scores, key=scores.get)
        confidence = min(scores[intent] / 3.0, 1.0) if scores[intent] > 0 else 0.5

        city = self._extract_city(query)
        attraction = self._extract_attraction(query)

        result = {
            'intent': intent,
            'confidence': confidence,
            'city': city,
            'attraction': attraction,
            'keywords': [],
            'method': 'rule_based'
        }

        logger.info(f"规则分类结果: intent={intent}, confidence={confidence}")
        return result

    def _extract_city(self, query: str) -> str:
        """
        从查询中提取城市名称

        Args:
            query: 查询文本

        Returns:
            城市名称
        """
        cities = ['三亚', '云南', '厦门', '张家界', '成都', '杭州']

        for city in cities:
            if city in query:
                return city

        return ''

    def _extract_attraction(self, query: str) -> str:
        """
        从查询中提取景点名称

        Args:
            query: 查询文本

        Returns:
            景点名称
        """
        attractions = [
            '天涯海角', '南山', '蜈支洲岛', '亚龙湾', '海棠湾',
            '大东海', '三亚湾', '鹿回头', '西岛'
        ]

        for attraction in attractions:
            if attraction in query:
                return attraction

        return ''
