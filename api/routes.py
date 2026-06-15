"""
API路由定义
处理HTTP请求
"""

import os
import time
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from config.settings import settings
from config.logging_config import get_logger
from data.document_parser import DocumentParser
from data.chunk_processor import ChunkProcessor
from data.embedding_service import EmbeddingService
from storage.milvus_client import MilvusClient
from intent.classifier import IntentClassifier
from retrieval.vector_search import VectorSearch
from retrieval.hybrid_search import HybridSearch
from generator.answer_generator import AnswerGenerator
from dialogue.session_manager import SessionManager
from dialogue.history_store import HistoryStore
from .schemas import (
    ImportResponse, QueryRequest, QueryResponse,
    StatsResponse, IntentResponse, SearchRequest, SearchResponse, SearchResult
)

logger = get_logger('api_routes')

router = APIRouter()

# 初始化服务实例
session_manager = SessionManager()
history_store = HistoryStore()
intent_classifier = IntentClassifier()
hybrid_search = HybridSearch()
answer_generator = AnswerGenerator()


@router.post("/api/import/files", response_model=ImportResponse)
async def import_files(files: List[UploadFile] = File(...)):
    """
    上传并导入Markdown文件

    Args:
        files: 上传的文件列表

    Returns:
        导入结果
    """
    start_time = time.time()

    try:
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        saved_files = []
        for file in files:
            if not file.filename.endswith('.md'):
                continue

            file_path = os.path.join(upload_dir, file.filename)
            content = await file.read()

            with open(file_path, 'wb') as f:
                f.write(content)

            saved_files.append(file_path)
            logger.info(f"保存上传文件: {file.filename}")

        if not saved_files:
            return ImportResponse(
                success=False,
                message="没有找到有效的Markdown文件"
            )

        parser = DocumentParser()
        processor = ChunkProcessor()
        embedding_service = EmbeddingService()
        milvus_client = MilvusClient()

        all_chunks = []
        for file_path in saved_files:
            chunks = parser.parse_file(file_path)
            all_chunks.extend(chunks)

        processed_chunks = processor.process_chunks(all_chunks)

        texts = [chunk.content for chunk in processed_chunks]
        vectors = embedding_service.embed_texts(texts)

        chunks_data = [chunk.to_dict() for chunk in processed_chunks]
        ids = milvus_client.insert_chunks(chunks_data, vectors)

        elapsed_time = time.time() - start_time

        stats = milvus_client.get_collection_stats()

        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass

        return ImportResponse(
            success=True,
            message=f"成功导入 {len(saved_files)} 个文件",
            total_files=len(saved_files),
            total_chunks=len(processed_chunks),
            inserted_count=len(ids),
            elapsed_time=elapsed_time
        )

    except Exception as e:
        logger.error(f"导入文件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/import/directory", response_model=ImportResponse)
async def import_directory():
    """
    从数据目录导入所有Markdown文件

    Returns:
        导入结果
    """
    start_time = time.time()

    try:
        parser = DocumentParser()
        processor = ChunkProcessor()
        embedding_service = EmbeddingService()
        milvus_client = MilvusClient()

        data_dir = settings.DATA_DIR

        if not os.path.exists(data_dir):
            return ImportResponse(
                success=False,
                message=f"数据目录不存在: {data_dir}"
            )

        chunks = parser.parse_directory(data_dir)
        processed_chunks = processor.process_chunks(chunks)

        texts = [chunk.content for chunk in processed_chunks]
        vectors = embedding_service.embed_texts(texts)

        chunks_data = [chunk.to_dict() for chunk in processed_chunks]
        ids = milvus_client.insert_chunks(chunks_data, vectors)

        elapsed_time = time.time() - start_time

        return ImportResponse(
            success=True,
            message="成功导入数据目录",
            total_files=len(set(c.source_filename for c in processed_chunks)),
            total_chunks=len(processed_chunks),
            inserted_count=len(ids),
            elapsed_time=elapsed_time
        )

    except Exception as e:
        logger.error(f"导入目录失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    智能问答

    Args:
        request: 查询请求

    Returns:
        查询响应
    """
    try:
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session()

        intent_result = intent_classifier.classify(request.query)
        intent = intent_result.get('intent', '知识问答')
        city = intent_result.get('city', '')

        filters = {}
        if city:
            filters['city'] = city

        search_results = hybrid_search.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters if filters else None
        )

        if not search_results:
            answer = "抱歉，我没有找到相关的旅游信息。您可以尝试换一种问法。"
        else:
            answer = answer_generator.generate(
                query=request.query,
                search_results=search_results,
                intent=intent
            )

        history_store.add_user_message(session_id, request.query, {
            'intent': intent,
            'city': city
        })
        history_store.add_assistant_message(session_id, answer)
        history_store.save_history(session_id)

        session_manager.increment_message_count(session_id)

        sources = []
        for result in search_results[:3]:
            sources.append({
                'title': result.get('section_title', ''),
                'city': result.get('city', ''),
                'source': result.get('source_filename', ''),
                'score': result.get('score', 0)
            })

        return QueryResponse(
            success=True,
            answer=answer,
            intent=intent,
            search_results_count=len(search_results),
            session_id=session_id,
            sources=sources
        )

    except Exception as e:
        logger.error(f"查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    获取知识库统计信息

    Returns:
        统计信息
    """
    try:
        milvus_client = MilvusClient()
        stats = milvus_client.get_collection_stats()

        return StatsResponse(
            success=True,
            collection_name=stats['collection_name'],
            num_entities=stats['num_entities'],
            is_loaded=not stats['is_loaded']
        )

    except Exception as e:
        logger.error(f"获取统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/intent", response_model=IntentResponse)
async def classify_intent(query: str):
    """
    意图识别

    Args:
        query: 用户问题

    Returns:
        意图识别结果
    """
    try:
        result = intent_classifier.classify(query)

        return IntentResponse(
            success=True,
            intent=result.get('intent', ''),
            confidence=result.get('confidence', 0),
            city=result.get('city', ''),
            attraction=result.get('attraction', ''),
            keywords=result.get('keywords', [])
        )

    except Exception as e:
        logger.error(f"意图识别失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    向量搜索

    Args:
        request: 搜索请求

    Returns:
        搜索结果
    """
    try:
        vector_search = VectorSearch()

        results = vector_search.search(
            query=request.query,
            top_k=request.top_k,
            content_type_filter=request.content_type,
            city_filter=request.city
        )

        search_results = []
        for result in results:
            search_results.append(SearchResult(
                content=result.get('content', ''),
                section_title=result.get('section_title', ''),
                city=result.get('city', ''),
                attraction_name=result.get('attraction_name', ''),
                content_type=result.get('content_type', ''),
                source_filename=result.get('source_filename', ''),
                score=result.get('score', 0)
            ))

        return SearchResponse(
            success=True,
            results=search_results,
            total_count=len(search_results)
        )

    except Exception as e:
        logger.error(f"搜索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
