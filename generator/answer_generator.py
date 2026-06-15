"""
答案生成器
整合检索结果并生成最终回答
"""

from typing import List, Dict, Any, Optional, Generator
from .llm_client import LLMClient
from .prompts import ANSWER_GENERATION_PROMPT, INTENT_SPECIFIC_PROMPTS
from config.logging_config import get_logger

logger = get_logger('answer_generator')


class AnswerGenerator:
    """答案生成器"""

    def __init__(self):
        self.llm_client = LLMClient()

    def generate(self, query: str,
                 search_results: List[Dict[str, Any]],
                 intent: str = '知识问答') -> str:
        """
        生成回答

        Args:
            query: 用户查询
            search_results: 检索结果列表
            intent: 意图类型

        Returns:
            生成的回答文本
        """
        if not search_results:
            return self._generate_no_result_response(query)

        context = self._format_context(search_results)

        prompt = ANSWER_GENERATION_PROMPT.format(
            context=context,
            query=query
        )

        intent_prompt = INTENT_SPECIFIC_PROMPTS.get(intent, '')
        if intent_prompt:
            prompt += intent_prompt

        try:
            answer = self.llm_client.generate_answer(prompt)

            logger.info(f"答案生成完成: query='{query}', 长度={len(answer)}")
            return answer

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_stream(self, query: str,
                        search_results: List[Dict[str, Any]],
                        intent: str = '知识问答') -> Generator[str, None, None]:
        """
        流式生成回答

        Args:
            query: 用户查询
            search_results: 检索结果列表
            intent: 意图类型

        Yields:
            流式返回的回答片段
        """
        if not search_results:
            yield self._generate_no_result_response(query)
            return

        context = self._format_context(search_results)

        prompt = ANSWER_GENERATION_PROMPT.format(
            context=context,
            query=query
        )

        intent_prompt = INTENT_SPECIFIC_PROMPTS.get(intent, '')
        if intent_prompt:
            prompt += intent_prompt

        try:
            messages = [
                {"role": "system", "content": "你是一个专业的旅游智能助手。"},
                {"role": "user", "content": prompt}
            ]

            for chunk in self.llm_client.chat_stream(messages):
                yield chunk

        except Exception as e:
            logger.error(f"流式答案生成失败: {e}")
            yield f"\n\n抱歉，生成回答时出现错误：{str(e)}"

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化检索结果为上下文

        Args:
            results: 检索结果列表

        Returns:
            格式化的上下文字符串
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            section_title = result.get('section_title', '')
            content = result.get('content', '')
            city = result.get('city', '')
            attraction = result.get('attraction_name', '')
            source = result.get('source_filename', '')
            score = result.get('score', result.get('final_score', 0))

            context_part = f"【参考资料 {i}】"
            if section_title:
                context_part += f"\n标题：{section_title}"
            if city:
                context_part += f"\n城市：{city}"
            if attraction:
                context_part += f"\n景点：{attraction}"
            if source:
                context_part += f"\n来源：{source}"
            context_part += f"\n相似度：{score:.3f}"
            context_part += f"\n内容：\n{content}\n"

            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _generate_no_result_response(self, query: str) -> str:
        """
        生成无检索结果时的回复

        Args:
            query: 用户查询

        Returns:
            回复文本
        """
        return f"""抱歉，我暂时没有找到与"{query}"相关的旅游信息。

可能的原因：
1. 您的问题可能超出了当前知识库的范围
2. 可以尝试换一种问法
3. 或者询问其他城市、景点的相关信息

您可以尝试询问：
- "三亚有哪些必去景点？"
- "帮我规划一个3日游行程"
- "天涯海角门票多少钱？"

我会尽力为您提供帮助！"""

    def refine_answer(self, answer: str) -> str:
        """
        优化回答质量

        Args:
            answer: 原始回答

        Returns:
            优化后的回答
        """
        from .prompts import REFINEMENT_PROMPT

        prompt = REFINEMENT_PROMPT.format(answer=answer)

        try:
            refined = self.llm_client.generate_answer(prompt)
            logger.info(f"回答优化完成")
            return refined
        except Exception as e:
            logger.warning(f"回答优化失败，返回原始回答: {e}")
            return answer

    def format_attraction_list(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化为景点列表

        Args:
            results: 检索结果

        Returns:
            格式化的列表文本
        """
        if not results:
            return "未找到相关景点信息。"

        lines = ["为您找到以下景点推荐：\n"]

        for i, result in enumerate(results, 1):
            title = result.get('section_title', '')
            attraction = result.get('attraction_name', '')
            city = result.get('city', '')
            content = result.get('content', '')[:200]

            line = f"{i}. **{title or attraction}**"
            if city:
                line += f" ({city})"
            if content:
                line += f"\n   {content}..."

            lines.append(line)

        return "\n\n".join(lines)

    def format_route_plan(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化为行程计划

        Args:
            results: 检索结果

        Returns:
            格式化的行程文本
        """
        if not results:
            return "未找到相关线路信息。"

        lines = ["为您推荐的行程安排：\n"]

        for i, result in enumerate(results, 1):
            title = result.get('section_title', '')
            content = result.get('content', '')
            source = result.get('source_filename', '')

            lines.append(f"【方案 {i}】{title}")
            lines.append(content)
            if source:
                lines.append(f"*来源：{source}*")
            lines.append("")

        return "\n".join(lines)
