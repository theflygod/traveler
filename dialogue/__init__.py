"""
对话管理模块
管理会话状态和历史记录
"""

from .session_manager import SessionManager
from .history_store import HistoryStore

__all__ = ['SessionManager', 'HistoryStore']
