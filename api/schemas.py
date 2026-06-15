"""
Pydantic数据模型
定义API请求和响应的数据结构
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ImportRequest(BaseModel):
    """导入请求"""
    force_reimport: bool = Field(False, description="是否强制重新导入")


class ImportResponse(BaseModel):
    """导入响应"""
    success: bool
    message: str
    total_files: int = 0
    total_chunks: int = 0
    inserted_count: int = 0
    elapsed_time: float = 0.0


class QueryRequest(BaseModel):
    """查询请求"""
    query: str
    session_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    """查询响应"""
    success: bool
    answer: str
    intent: str = ""
    search_results_count: int = 0
    session_id: str = ""
    sources: List[Dict[str, Any]] = []


class StatsResponse(BaseModel):
    """统计信息响应"""
    success: bool
    collection_name: str = ""
    num_entities: int = 0
    is_loaded: bool = False


class IntentResponse(BaseModel):
    """意图识别响应"""
    success: bool
    intent: str = ""
    confidence: float = 0.0
    city: str = ""
    attraction: str = ""
    keywords: List[str] = []


class SearchResult(BaseModel):
    """搜索结果"""
    content: str
    section_title: str = ""
    city: str = ""
    attraction_name: str = ""
    content_type: str = ""
    source_filename: str = ""
    score: float = 0.0


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    top_k: int = Field(5, ge=1, le=20)
    content_type: Optional[str] = None
    city: Optional[str] = None


class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool
    results: List[SearchResult] = []
    total_count: int = 0
