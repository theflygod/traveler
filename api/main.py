"""
FastAPI 主应用
提供Web API服务和前端页面
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import uvicorn
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema, Collection, utility
from config.settings import settings
from config.logging_config import setup_logging, get_logger
from workflow.graph import TravelKnowledgeGraph
from dialogue.session_manager import SessionManager
from dialogue.history_store import HistoryStore

logger = setup_logging(log_level='INFO')

app = FastAPI(
    title="旅游知识库智能助手",
    description="基于LangGraph的智能旅游问答系统",
    version="1.0.0"
)

travel_graph = TravelKnowledgeGraph()
session_manager = SessionManager()
history_store = HistoryStore()

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """查询响应模型"""
    success: bool
    query: str
    answer: str
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    extracted_entities: Optional[dict] = None
    search_results: Optional[list] = None
    error: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def home():
    """首页"""
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """聊天接口"""
    try:
        session_id = request.session_id or session_manager.create_session()

        result = travel_graph.run(
            query=request.query,
            session_id=session_id
        )

        history_store.add_user_message(session_id, request.query)
        history_store.add_assistant_message(session_id, result.get('answer', ''))
        history_store.save_history(session_id)

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"聊天接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream")
async def stream_chat(request: QueryRequest):
    """流式聊天接口"""
    from fastapi.responses import StreamingResponse
    import json

    async def generate():
        session_id = request.session_id or session_manager.create_session()

        for event in travel_graph.run_stream(
                query=request.query,
                session_id=session_id
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/session/new")
async def create_session():
    """创建新会话"""
    session_id = session_manager.create_session()
    return {"session_id": session_id}


@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    from storage.milvus_client import MilvusClient

    try:
        milvus_client = MilvusClient()
        stats = milvus_client.get_collection_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
