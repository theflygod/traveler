"""
会话管理器
管理用户会话状态和上下文
"""

import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger('session_manager')


class Session:
    """会话对象"""

    def __init__(self, session_id: str, user_id: str = None):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.context = {}
        self.message_count = 0

    def update_activity(self):
        """更新最后活动时间"""
        self.last_active = datetime.now()

    def add_context(self, key: str, value: Any):
        """添加上下文信息"""
        self.context[key] = value
        self.update_activity()

    def get_context(self, key: str, default=None) -> Any:
        """获取上下文信息"""
        return self.context.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'message_count': self.message_count,
            'context': self.context
        }


class SessionManager:
    """会话管理器（单例模式）"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.sessions: Dict[str, Session] = {}
        self.max_sessions = 1000
        self.session_timeout = 3600

        self._initialized = True
        logger.info("会话管理器初始化完成")

    def create_session(self, user_id: str = None) -> str:
        """
        创建新会话

        Args:
            user_id: 用户ID（可选）

        Returns:
            会话ID
        """
        session_id = str(uuid.uuid4())
        session = Session(session_id, user_id)

        if len(self.sessions) >= self.max_sessions:
            self._cleanup_expired_sessions()

        self.sessions[session_id] = session

        logger.info(f"创建新会话: session_id={session_id[:8]}...")
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话

        Args:
            session_id: 会话ID

        Returns:
            会话对象，不存在则返回None
        """
        session = self.sessions.get(session_id)

        if session:
            if self._is_session_expired(session):
                self.remove_session(session_id)
                return None
            session.update_activity()

        return session

    def remove_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除会话: session_id={session_id[:8]}...")
            return True
        return False

    def increment_message_count(self, session_id: str):
        """
        增加消息计数

        Args:
            session_id: 会话ID
        """
        session = self.get_session(session_id)
        if session:
            session.message_count += 1

    def set_session_context(self, session_id: str, key: str, value: Any):
        """
        设置会话上下文

        Args:
            session_id: 会话ID
            key: 上下文键
            value: 上下文值
        """
        session = self.get_session(session_id)
        if session:
            session.add_context(key, value)

    def get_session_context(self, session_id: str, key: str, default=None) -> Any:
        """
        获取会话上下文

        Args:
            session_id: 会话ID
            key: 上下文键
            default: 默认值

        Returns:
            上下文值
        """
        session = self.get_session(session_id)
        if session:
            return session.get_context(key, default)
        return default

    def _is_session_expired(self, session: Session) -> bool:
        """
        检查会话是否过期

        Args:
            session: 会话对象

        Returns:
            是否过期
        """
        elapsed = (datetime.now() - session.last_active).total_seconds()
        return elapsed > self.session_timeout

    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        expired_ids = [
            sid for sid, session in self.sessions.items()
            if self._is_session_expired(session)
        ]

        for sid in expired_ids:
            del self.sessions[sid]

        if expired_ids:
            logger.info(f"清理{len(expired_ids)}个过期会话")

    def get_active_session_count(self) -> int:
        """获取活跃会话数"""
        return len(self.sessions)

    def clear_all_sessions(self):
        """清空所有会话"""
        self.sessions.clear()
        logger.info("已清空所有会话")
