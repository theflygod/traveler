"""
向量化服务
使用Embedding模型将文本转换为向量
"""

import os
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    """向量化服务"""

    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.api_base = os.getenv('OPENAI_API_BASE')
        self.model = os.getenv('EMBEDDING_MODEL', 'text-embedding-v4')
        self.dimension = int(os.getenv('EMBEDDING_DIM', '1536'))

        if not self.api_key:
            raise ValueError("未配置 OPENAI_API_KEY 环境变量")

        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

    def embed_text(self, text: str) -> List[float]:
        """
        将单个文本转换为向量

        Args:
            text: 输入文本

        Returns:
            向量表示
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )

            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            raise Exception(f"向量化失败: {str(e)}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if not texts:
            return []

        embeddings = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"批量向量化失败 (批次 {i}): {str(e)}")
                for text in batch:
                    try:
                        embedding = self.embed_text(text)
                        embeddings.append(embedding)
                    except Exception as e2:
                        print(f"单条向量化也失败: {str(e2)}")
                        embeddings.append([0.0] * self.dimension)

        return embeddings

    def get_dimension(self) -> int:
        """
        获取向量维度

        Returns:
            向量维度
        """
        return self.dimension
