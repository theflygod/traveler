"""
LLM客户端
封装与大语言模型的交互
"""

from typing import List, Dict, Any, Optional, Generator
from openai import OpenAI
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger('llm_client')


class LLMClient:
    """LLM客户端（单例模式）"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        self.model = settings.LLM_DEFAULT_MODEL
        self.temperature = settings.LLM_DEFAULT_TEMPERATURE

        self._initialized = True
        logger.info(f"LLM客户端初始化完成: model={self.model}")

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = None,
             max_tokens: int = 1000) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLM响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM调用成功，返回{len(answer)}字符")
            return answer

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise

    def chat_stream(self, messages: List[Dict[str, str]],
                    temperature: float = None,
                    max_tokens: int = 1000) -> Generator[str, None, None]:
        """
        发送流式聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Yields:
            流式返回的文本片段
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM流式调用失败: {e}")
            raise

    def generate_answer(self, prompt: str,
                        system_message: str = None,
                        temperature: float = None) -> str:
        """
        生成回答（简化接口）

        Args:
            prompt: 用户提示
            system_message: 系统消息
            temperature: 温度参数

        Returns:
            生成的回答
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, temperature)
