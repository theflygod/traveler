"""
旅游知识库系统 - 模块测试脚本
测试各个核心模块的功能
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import settings
from config.logging_config import setup_logging, get_logger
from data.document_parser import DocumentParser
from data.chunk_processor import ChunkProcessor
from data.metadata_extractor import MetadataExtractor
from data.embedding_service import EmbeddingService
from storage.milvus_client import MilvusClient
from intent.classifier import IntentClassifier
from retrieval.vector_search import VectorSearch
from retrieval.hybrid_search import HybridSearch
from generator.answer_generator import AnswerGenerator
from dialogue.session_manager import SessionManager
from dialogue.history_store import HistoryStore

logger = None


def print_section(title: str):
    """打印测试章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(success: bool, message: str):
    """打印测试结果"""
    icon = "✅" if success else "❌"
    status = "通过" if success else "失败"
    print(f"{icon} [{status}] {message}")


def test_config_module():
    """测试配置模块"""
    print_section("测试1: 配置模块 (config)")

    try:
        print(f"\n📋 配置信息:")
        print(f"   LLM模型: {settings.LLM_DEFAULT_MODEL}")
        print(f"   LLM温度: {settings.LLM_DEFAULT_TEMPERATURE}")
        print(f"   API地址: {settings.OPENAI_API_BASE}")
        print(f"   Embedding模型: {settings.EMBEDDING_MODEL}")
        print(f"   向量维度: {settings.EMBEDDING_DIM}")
        print(f"   Milvus地址: {settings.MILVUS_URL}")
        print(f"   集合名称: {settings.CHUNKS_COLLECTION}")
        print(f"   数据目录: {settings.DATA_DIR}")

        config_dict = settings.to_dict()
        assert 'LLM' in config_dict
        assert 'Embedding' in config_dict
        assert 'Milvus' in config_dict

        print_result(True, "配置加载成功")
        return True

    except Exception as e:
        print_result(False, f"配置加载失败: {e}")
        logger.error(f"配置模块测试失败: {e}", exc_info=True)
        return False


def test_logging_module():
    """测试日志模块"""
    print_section("测试2: 日志模块 (logging)")

    try:
        test_logger = get_logger('test')
        test_logger.info("这是一条测试日志")
        test_logger.debug("这是一条调试日志")

        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        exists = os.path.exists(log_dir)

        print(f"\n📝 日志目录: {log_dir}")
        print(f"   目录存在: {exists}")

        print_result(True, "日志模块工作正常")
        return True

    except Exception as e:
        print_result(False, f"日志模块失败: {e}")
        logger.error(f"日志模块测试失败: {e}", exc_info=True)
        return False


def test_document_parser():
    """测试文档解析器"""
    print_section("测试3: 文档解析器 (document_parser)")

    try:
        parser = DocumentParser()

        test_file = os.path.join(settings.DATA_DIR, '景点攻略', '三亚景点推荐.md')

        if not os.path.exists(test_file):
            print(f"\n⚠️  测试文件不存在: {test_file}")
            print_result(False, "跳过测试")
            return False

        print(f"\n📄 测试文件: {test_file}")

        start_time = time.time()
        chunks = parser.parse_file(test_file)
        elapsed = time.time() - start_time

        print(f"   分块数量: {len(chunks)}")
        print(f"   处理时间: {elapsed:.2f}秒")

        if chunks:
            first_chunk = chunks[0]
            print(f"\n   第一个分块示例:")
            print(f"   - 标题: {first_chunk.section_title[:50]}")
            print(f"   - 内容长度: {len(first_chunk.content)}字符")
            print(f"   - 文件名: {first_chunk.source_filename}")
            print(f"   - 内容预览: {first_chunk.content[:100]}...")

        assert len(chunks) > 0, "没有解析出任何分块"

        print_result(True, f"文档解析成功，共{len(chunks)}个分块")
        return True

    except Exception as e:
        print_result(False, f"文档解析失败: {e}")
        logger.error(f"文档解析器测试失败: {e}", exc_info=True)
        return False


def test_metadata_extractor():
    """测试元数据提取器"""
    print_section("测试4: 元数据提取器 (metadata_extractor)")

    try:
        parser = DocumentParser()
        extractor = MetadataExtractor()

        test_file = os.path.join(settings.DATA_DIR, '景点攻略', '三亚景点推荐.md')

        if not os.path.exists(test_file):
            print(f"\n⚠️  测试文件不存在")
            print_result(False, "跳过测试")
            return False

        chunks = parser.parse_file(test_file)

        print(f"\n🔍 提取元数据示例:\n")
        for i, chunk in enumerate(chunks[:3], 1):
            processed_chunk = extractor.extract_metadata(chunk)

            print(f"分块 {i}:")
            print(f"   标题: {processed_chunk.section_title}")
            print(f"   内容类型: {processed_chunk.content_type}")
            print(f"   城市: {processed_chunk.city}")
            print(f"   景点: {processed_chunk.attraction_name}")
            print()

        print_result(True, "元数据提取成功")
        return True

    except Exception as e:
        print_result(False, f"元数据提取失败: {e}")
        logger.error(f"元数据提取器测试失败: {e}", exc_info=True)
        return False


def test_chunk_processor():
    """测试分块处理器"""
    print_section("测试5: 分块处理器 (chunk_processor)")

    try:
        processor = ChunkProcessor()

        test_file = os.path.join(settings.DATA_DIR, '景点攻略', '三亚景点推荐.md')

        if not os.path.exists(test_file):
            print(f"\n⚠️  测试文件不存在")
            print_result(False, "跳过测试")
            return False

        parser = DocumentParser()
        raw_chunks = parser.parse_file(test_file)

        print(f"\n⚙️  处理分块...")
        print(f"   原始分块数: {len(raw_chunks)}")

        start_time = time.time()
        processed_chunks = processor.process_chunks(raw_chunks)
        elapsed = time.time() - start_time

        print(f"   处理后分块数: {len(processed_chunks)}")
        print(f"   处理时间: {elapsed:.2f}秒")

        if processed_chunks:
            chunk = processed_chunks[0]
            print(f"\n   处理后示例:")
            print(f"   - 标题: {chunk.section_title}")
            print(f"   - 内容类型: {chunk.content_type}")
            print(f"   - 城市: {chunk.city}")
            print(f"   - 内容长度: {len(chunk.content)}字符")

        print_result(True, "分块处理成功")
        return True

    except Exception as e:
        print_result(False, f"分块处理失败: {e}")
        logger.error(f"分块处理器测试失败: {e}", exc_info=True)
        return False


def test_embedding_service():
    """测试向量化服务"""
    print_section("测试6: 向量化服务 (embedding_service)")

    try:
        embedding_service = EmbeddingService()

        test_texts = [
            "三亚是一个美丽的海滨城市",
            "亚龙湾是三亚最著名的海湾之一"
        ]

        print(f"\n🔢 测试向量化:")
        print(f"   模型: {embedding_service.model}")
        print(f"   维度: {embedding_service.get_dimension()}")

        start_time = time.time()
        vector = embedding_service.embed_text(test_texts[0])
        elapsed = time.time() - start_time

        print(f"\n   单条向量化:")
        print(f"   - 文本: {test_texts[0]}")
        print(f"   - 向量长度: {len(vector)}")
        print(f"   - 向量前5维: {vector[:5]}")
        print(f"   - 耗时: {elapsed:.2f}秒")

        start_time = time.time()
        vectors = embedding_service.embed_texts(test_texts)
        elapsed = time.time() - start_time

        print(f"\n   批量向量化:")
        print(f"   - 文本数量: {len(test_texts)}")
        print(f"   - 向量数量: {len(vectors)}")
        print(f"   - 耗时: {elapsed:.2f}秒")

        assert len(vector) == embedding_service.get_dimension()
        assert len(vectors) == len(test_texts)

        print_result(True, "向量化服务正常")
        return True

    except Exception as e:
        print_result(False, f"向量化服务失败: {e}")
        logger.error(f"向量化服务测试失败: {e}", exc_info=True)
        return False


def test_milvus_client():
    """测试Milvus客户端"""
    print_section("测试7: Milvus客户端 (milvus_client)")

    try:
        milvus_client = MilvusClient()

        print(f"\n💾 连接信息:")
        print(f"   Milvus地址: {settings.MILVUS_URL}")
        print(f"   集合名称: {settings.CHUNKS_COLLECTION}")

        stats = milvus_client.get_collection_stats()

        print(f"\n📊 集合统计:")
        print(f"   记录数: {stats['num_entities']}")
        print(f"   加载状态: {'已加载' if not stats['is_loaded'] else '未加载'}")

        print_result(True, "Milvus客户端连接成功")
        return True

    except Exception as e:
        print_result(False, f"Milvus客户端失败: {e}")
        logger.error(f"Milvus客户端测试失败: {e}", exc_info=True)
        return False


def test_intent_classifier():
    """测试意图分类器"""
    print_section("测试8: 意图分类器 (intent_classifier)")

    try:
        classifier = IntentClassifier()

        test_queries = [
            "三亚有哪些好玩的景点？",
            "帮我规划一个三亚3日游行程",
            "天涯海角的门票多少钱？",
            "去三亚旅游需要注意什么？"
        ]

        print(f"\n🧠 测试意图识别:\n")

        for query in test_queries:
            print(f"问题: {query}")

            start_time = time.time()
            result = classifier.classify(query)
            elapsed = time.time() - start_time

            print(f"   意图: {result.get('intent')}")
            print(f"   置信度: {result.get('confidence', 0):.2f}")
            print(f"   城市: {result.get('city', 'N/A')}")
            print(f"   景点: {result.get('attraction', 'N/A')}")
            print(f"   耗时: {elapsed:.2f}秒")
            print()

        print_result(True, "意图分类器工作正常")
        return True

    except Exception as e:
        print_result(False, f"意图分类器失败: {e}")
        logger.error(f"意图分类器测试失败: {e}", exc_info=True)
        return False


def test_vector_search():
    """测试向量检索"""
    print_section("测试9: 向量检索 (vector_search)")

    try:
        vector_search = VectorSearch()

        test_query = "三亚有什么好玩的景点"

        print(f"\n🔍 测试查询: {test_query}\n")

        start_time = time.time()
        results = vector_search.search_attractions(test_query, top_k=3)
        elapsed = time.time() - start_time

        print(f"   返回结果数: {len(results)}")
        print(f"   耗时: {elapsed:.2f}秒")

        if results:
            print(f"\n   检索结果示例:\n")
            for i, result in enumerate(results[:2], 1):
                print(f"结果 {i}:")
                print(f"   标题: {result.get('section_title', 'N/A')[:50]}")
                print(f"   城市: {result.get('city', 'N/A')}")
                print(f"   相似度: {result.get('score', 0):.3f}")
                print(f"   内容预览: {result.get('content', '')[:100]}...")
                print()

        print_result(True, "向量检索成功")
        return True

    except Exception as e:
        print_result(False, f"向量检索失败: {e}")
        logger.error(f"向量检索测试失败: {e}", exc_info=True)
        return False


def test_hybrid_search():
    """测试混合检索"""
    print_section("测试10: 混合检索 (hybrid_search)")

    try:
        hybrid_search = HybridSearch()

        test_query = "三亚天涯海角门票价格"

        print(f"\n🔍 测试查询: {test_query}\n")

        start_time = time.time()
        results = hybrid_search.search(test_query, top_k=3)
        elapsed = time.time() - start_time

        print(f"   返回结果数: {len(results)}")
        print(f"   耗时: {elapsed:.2f}秒")

        if results:
            print(f"\n   检索结果示例:\n")
            for i, result in enumerate(results[:2], 1):
                print(f"结果 {i}:")
                print(f"   标题: {result.get('section_title', 'N/A')[:50]}")
                print(f"   分数: {result.get('final_score', 0):.3f}")
                print(f"   内容预览: {result.get('content', '')[:100]}...")
                print()

        print_result(True, "混合检索成功")
        return True

    except Exception as e:
        print_result(False, f"混合检索失败: {e}")
        logger.error(f"混合检索测试失败: {e}", exc_info=True)
        return False


def test_answer_generator():
    """测试答案生成器"""
    print_section("测试11: 答案生成器 (answer_generator)")

    try:
        answer_generator = AnswerGenerator()
        vector_search = VectorSearch()

        test_query = "三亚有哪些必去景点？"

        print(f"\n💬 测试问题: {test_query}\n")

        print("🔍 正在检索相关资料...")
        search_results = vector_search.search_attractions(test_query, top_k=3)

        if not search_results:
            print("   ⚠️  未找到相关资料，请先导入数据")
            print_result(True, "跳过测试（知识库为空，需先执行import）")
            return True

        print(f"   ✅ 找到 {len(search_results)} 条资料\n")

        print("🤖 生成回答中...\n")
        start_time = time.time()
        answer = answer_generator.generate(
            query=test_query,
            search_results=search_results,
            intent='景点介绍'
        )
        elapsed = time.time() - start_time

        print(f"回答:\n{answer}\n")
        print(f"   耗时: {elapsed:.2f}秒")
        print(f"   回答长度: {len(answer)}字符")

        print_result(True, "答案生成成功")
        return True

    except Exception as e:
        print_result(False, f"答案生成失败: {e}")
        logger.error(f"答案生成器测试失败: {e}", exc_info=True)
        return False


def test_session_manager():
    """测试会话管理器"""
    print_section("测试12: 会话管理器 (session_manager)")

    try:
        session_manager = SessionManager()

        print(f"\n👥 测试会话管理:\n")

        session_id = session_manager.create_session(user_id='test_user')
        print(f"   创建会话: {session_id[:8]}...")

        session_manager.set_session_context(session_id, 'city', '三亚')
        city = session_manager.get_session_context(session_id, 'city')
        print(f"   设置上下文: city={city}")

        session_manager.increment_message_count(session_id)
        session_manager.increment_message_count(session_id)

        session = session_manager.get_session(session_id)
        print(f"   消息计数: {session.message_count}")
        print(f"   活跃会话数: {session_manager.get_active_session_count()}")

        print_result(True, "会话管理器工作正常")
        return True

    except Exception as e:
        print_result(False, f"会话管理器失败: {e}")
        logger.error(f"会话管理器测试失败: {e}", exc_info=True)
        return False


def test_history_store():
    """测试历史记录存储"""
    print_section("测试13: 历史记录存储 (history_store)")

    try:
        history_store = HistoryStore()

        print(f"\n📜 测试历史记录:\n")

        session_id = 'test_session_' + str(int(time.time()))

        history_store.add_user_message(session_id, "三亚有哪些景点？")
        history_store.add_assistant_message(session_id, "三亚有亚龙湾、蜈支洲岛等景点...")

        messages = history_store.get_conversation_messages(session_id, limit=10)
        print(f"   消息数量: {len(messages)}")

        for msg in messages:
            print(f"   - {msg['role']}: {msg['content'][:50]}...")

        history_store.save_history(session_id)

        loaded_history = history_store.load_history(session_id)
        print(f"   加载消息数: {len(loaded_history.messages)}")

        history_store.delete_history(session_id)

        print_result(True, "历史记录存储正常")
        return True

    except Exception as e:
        print_result(False, f"历史记录存储失败: {e}")
        logger.error(f"历史记录存储测试失败: {e}", exc_info=True)
        return False


def run_all_tests():
    """运行所有测试"""
    global logger

    logger = setup_logging(log_level='INFO')

    print("\n" + "🧪" * 35)
    print("  旅游知识库系统 - 模块测试")
    print("🧪" * 35)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("配置模块", test_config_module),
        ("日志模块", test_logging_module),
        ("文档解析器", test_document_parser),
        ("元数据提取器", test_metadata_extractor),
        ("分块处理器", test_chunk_processor),
        ("向量化服务", test_embedding_service),
        ("Milvus客户端", test_milvus_client),
        ("意图分类器", test_intent_classifier),
        ("向量检索", test_vector_search),
        ("混合检索", test_hybrid_search),
        ("答案生成器", test_answer_generator),
        ("会话管理器", test_session_manager),
        ("历史记录存储", test_history_store),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print_result(False, f"{test_name}异常: {e}")
            logger.error(f"{test_name}测试异常: {e}", exc_info=True)
            results.append((test_name, False))

    print_section("测试结果汇总")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n📊 测试统计:\n")
    for test_name, success in results:
        icon = "✅" if success else "❌"
        status = "通过" if success else "失败"
        print(f"   {icon} {test_name}: {status}")

    print(f"\n{'=' * 70}")
    print(f"  总计: {passed}/{total} 通过")
    print(f"  通过率: {passed / total * 100:.1f}%")
    print(f"{'=' * 70}")

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n结束时间: {end_time}\n")

    if passed == total:
        print("🎉 所有测试通过！\n")
    else:
        print(f"⚠️  有 {total - passed} 个测试失败，请检查日志\n")

    return passed == total


if __name__ == '__main__':
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试执行异常: {e}\n")
        sys.exit(1)
