"""
历史记录存储
管理对话历史记录的保存和检索
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger('history_store')


class Message:
    """消息对象"""

    def __init__(self, role: str, content: str,
                 metadata: Dict[str, Any] = None):
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.timestamp = datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息对象"""
        msg = cls(
            role=data['role'],
            content=data['content'],
            metadata=data.get('metadata', {})
        )
        msg.timestamp = datetime.fromisoformat(data['timestamp'])
        return msg


class ConversationHistory:
    """对话历史"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Message] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_message(self, role: str, content: str,
                    metadata: Dict[str, Any] = None):
        """添加消息"""
        message = Message(role, content, metadata)
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_messages(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取消息列表

        Args:
            limit: 限制返回数量

        Returns:
            消息字典列表
        """
        messages = self.messages[-limit:] if limit else self.messages
        return [msg.to_dict() for msg in messages]

    def get_last_user_message(self) -> Optional[str]:
        """获取最后一条用户消息"""
        for msg in reversed(self.messages):
            if msg.role == 'user':
                return msg.content
        return None

    def clear(self):
        """清空历史"""
        self.messages.clear()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'messages': [msg.to_dict() for msg in self.messages],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'message_count': len(self.messages)
        }


class HistoryStore:
    """历史记录存储器"""

    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            project_root = os.path.dirname(os.path.dirname(__file__))
            storage_dir = os.path.join(project_root, 'data', 'history')

        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self.histories: Dict[str, ConversationHistory] = {}

        logger.info(f"历史记录存储器初始化完成: {storage_dir}")

    def get_history(self, session_id: str) -> ConversationHistory:
        """
        获取对话历史

        Args:
            session_id: 会话ID

        Returns:
            对话历史对象
        """
        if session_id not in self.histories:
            self.histories[session_id] = self._load_history(session_id)

        return self.histories[session_id]

    def add_user_message(self, session_id: str, content: str,
                         metadata: Dict[str, Any] = None):
        """
        添加用户消息

        Args:
            session_id: 会话ID
            content: 消息内容
            metadata: 元数据
        """
        history = self.get_history(session_id)
        history.add_message('user', content, metadata)
        logger.debug(f"添加用户消息: session_id={session_id[:8]}...")

    def add_assistant_message(self, session_id: str, content: str,
                              metadata: Dict[str, Any] = None):
        """
        添加助手消息

        Args:
            session_id: 会话ID
            content: 消息内容
            metadata: 元数据
        """
        history = self.get_history(session_id)
        history.add_message('assistant', content, metadata)
        logger.debug(f"添加助手消息: session_id={session_id[:8]}...")

    def get_conversation_messages(self, session_id: str,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取对话消息列表

        Args:
            session_id: 会话ID
            limit: 限制数量

        Returns:
            消息列表
        """
        history = self.get_history(session_id)
        return history.get_messages(limit)

    def save_history(self, session_id: str):
        """
        保存历史记录到文件

        Args:
            session_id: 会话ID
        """
        if session_id in self.histories:
            history = self.histories[session_id]
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(history.to_dict(), f, ensure_ascii=False, indent=2)

                logger.debug(f"保存历史记录: {session_id[:8]}...")
            except Exception as e:
                logger.error(f"保存历史记录失败: {e}")

    def load_history(self, session_id: str) -> ConversationHistory:
        """
        从文件加载历史记录

        Args:
            session_id: 会话ID

        Returns:
            对话历史对象
        """
        return self._load_history(session_id)

    def _load_history(self, session_id: str) -> ConversationHistory:
        """
        内部加载历史记录方法

        Args:
            session_id: 会话ID

        Returns:
            对话历史对象
        """
        file_path = os.path.join(self.storage_dir, f"{session_id}.json")

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                history = ConversationHistory(session_id)
                history.created_at = datetime.fromisoformat(data['created_at'])
                history.updated_at = datetime.fromisoformat(data['updated_at'])

                for msg_data in data['messages']:
                    msg = Message.from_dict(msg_data)
                    history.messages.append(msg)

                logger.debug(f"加载历史记录: {session_id[:8]}..., {len(history.messages)}条消息")
                return history

            except Exception as e:
                logger.error(f"加载历史记录失败: {e}")

        return ConversationHistory(session_id)

    def delete_history(self, session_id: str) -> bool:
        """
        删除历史记录

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        if session_id in self.histories:
            del self.histories[session_id]

        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"删除历史记录: {session_id[:8]}...")
                return True
            except Exception as e:
                logger.error(f"删除历史记录文件失败: {e}")

        return False

    def clear_all_histories(self):
        """清空所有历史记录"""
        self.histories.clear()

        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.storage_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"删除文件失败 {filename}: {e}")

        logger.info("已清空所有历史记录")

    def get_session_list(self) -> List[str]:
        """获取所有会话ID列表"""
        session_ids = []

        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                session_id = filename[:-5]
                session_ids.append(session_id)

        return session_ids
