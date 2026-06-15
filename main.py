"""
旅游知识库系统 - 主入口
支持命令行交互和Web服务两种模式
"""

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import settings
from config.logging_config import setup_logging, get_logger

logger = None


def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("🌴 旅游知识库智能助手 🌴")
    print("=" * 60)
    print("\n欢迎使用旅游知识库系统！")
    print("\n启动模式：")
    print("  1. web   - 启动Web服务（推荐，带图形界面）")
    print("  2. cli   - 命令行交互模式")
    print("  3. help  - 显示帮助")
    print("=" * 60 + "\n")


def print_help():
    """打印帮助信息"""
    print("\n【使用帮助】")
    print("\n启动方式：")
    print("  python main.py web  - 启动Web服务")
    print("  python main.py cli  - 启动命令行模式")
    print("\nWeb服务特性：")
    print("  ✓ 美观的图形界面")
    print("  ✓ 实时对话交互")
    print("  ✓ 快速提问按钮")
    print("  ✓ 自动会话管理")
    print("\n命令行模式命令：")
    print("  - import : 从files/数据目录导入文档到知识库")
    print("  - stats  : 查看知识库统计信息")
    print("  - clear  : 清空当前对话历史")
    print("  - help   : 显示此帮助信息")
    print("  - quit   : 退出系统")
    print("\n提示：")
    print("  - 首次使用前请先执行 import 命令导入数据")
    print("  - 支持多轮对话，系统会记住上下文")
    print("  - 回答结果会标注信息来源\n")


def import_documents():
    """导入文档到知识库"""
    print("\n📥 开始导入文档...\n")

    try:
        from data.document_parser import DocumentParser
        from data.chunk_processor import ChunkProcessor
        from data.embedding_service import EmbeddingService
        from storage.milvus_client import MilvusClient

        parser = DocumentParser()
        processor = ChunkProcessor()
        embedding_service = EmbeddingService()
        milvus_client = MilvusClient()

        data_dir = settings.DATA_DIR

        if not os.path.exists(data_dir):
            print(f"❌ 数据目录不存在: {data_dir}")
            return

        print(f"📂 扫描目录: {data_dir}")

        # 先列出所有要处理的文件
        all_md_files = []
        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.md'):
                    file_path = os.path.join(root, filename)
                    all_md_files.append(file_path)

        print(f"📄 找到 {len(all_md_files)} 个Markdown文件:")
        for i, file_path in enumerate(all_md_files, 1):
            rel_path = os.path.relpath(file_path, data_dir)
            print(f"   {i}. {rel_path}")
        print()

        chunks = parser.parse_directory(data_dir)
        print(f"✅ 解析完成，共 {len(chunks)} 个分块\n")

        # 统计每个文件的分块数
        file_chunk_count = {}
        for chunk in chunks:
            filename = chunk.source_filename
            file_chunk_count[filename] = file_chunk_count.get(filename, 0) + 1

        print("📊 各文件分块统计:")
        for filename, count in sorted(file_chunk_count.items()):
            print(f"   - {filename}: {count}个分块")
        print()

        print("⚙️  处理分块并提取元数据...")
        processed_chunks = processor.process_chunks(chunks)
        print(f"✅ 处理完成\n")

        print("🔢 生成向量并入库...")
        texts = [chunk.content for chunk in processed_chunks]
        vectors = embedding_service.embed_texts(texts)

        chunks_data = [chunk.to_dict() for chunk in processed_chunks]
        ids = milvus_client.insert_chunks(chunks_data, vectors)

        print(f"✅ 成功插入 {len(ids)} 条记录到Milvus\n")

        stats = milvus_client.get_collection_stats()
        print(f"📊 知识库统计:")
        print(f"   集合名称: {stats['collection_name']}")
        print(f"   记录总数: {stats['num_entities']}")
        print()

    except Exception as e:
        print(f"\n❌ 导入失败: {e}\n")
        if logger:
            logger.error(f"导入文档失败: {e}", exc_info=True)


def show_stats():
    """显示知识库统计"""
    print("\n📊 知识库统计信息\n")

    try:
        from storage.milvus_client import MilvusClient

        milvus_client = MilvusClient()
        stats = milvus_client.get_collection_stats()

        print(f"集合名称: {stats['collection_name']}")
        print(f"记录总数: {stats['num_entities']}")
        print(f"加载状态: {'已加载' if not stats['is_loaded'] else '未加载'}")
        print()

    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}\n")


def start_web():
    """启动Web服务"""
    print("\n🚀 正在启动Web服务...\n")
    print("📍 访问地址: http://localhost:8000")
    print("💡 在浏览器中打开上述地址即可使用")
    print("⚠️  按 Ctrl+C 停止服务\n")

    try:
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except ImportError:
        print("❌ 缺少依赖，请安装: pip install fastapi uvicorn jinja2 aiofiles")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        if logger:
            logger.error(f"Web服务启动失败: {e}", exc_info=True)


def start_cli():
    """启动命令行模式"""
    from data.document_parser import DocumentParser
    from data.chunk_processor import ChunkProcessor
    from data.embedding_service import EmbeddingService
    from storage.milvus_client import MilvusClient
    from intent.classifier import IntentClassifier
    from retrieval.hybrid_search import HybridSearch
    from generator.answer_generator import AnswerGenerator
    from dialogue.session_manager import SessionManager
    from dialogue.history_store import HistoryStore

    print("\n📝 进入命令行交互模式\n")
    print("提示：输入 'import' 导入数据，输入 'help' 查看帮助\n")

    session_manager = SessionManager()
    history_store = HistoryStore()
    intent_classifier = IntentClassifier()
    hybrid_search = HybridSearch()
    answer_generator = AnswerGenerator()

    session_id = session_manager.create_session()
    logger.info(f"创建会话: {session_id[:8]}...")

    while True:
        try:
            user_input = input("👤 您: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 感谢使用，再见！\n")
                logger.info("用户退出系统")
                break

            if user_input.lower() == 'help':
                print_help()
                continue

            if user_input.lower() == 'import':
                import_documents()
                continue

            if user_input.lower() == 'stats':
                show_stats()
                continue

            if user_input.lower() == 'clear':
                history_store.delete_history(session_id)
                session_id = session_manager.create_session()
                print("\n✅ 对话历史已清空\n")
                continue

            # 处理查询
            try:
                print("\n🤔 正在思考...\n")

                intent_result = intent_classifier.classify(user_input)
                intent = intent_result.get('intent', '知识问答')
                city = intent_result.get('city', '')
                attraction = intent_result.get('attraction', '')

                print(f"📋 意图识别: {intent}")
                if city:
                    print(f"📍 城市: {city}")
                if attraction:
                    print(f"🎯 景点: {attraction}")
                print()

                filters = {}
                if city:
                    filters['city'] = city

                search_results = hybrid_search.search(
                    query=user_input,
                    top_k=5,
                    filters=filters if filters else None
                )

                if not search_results:
                    answer = "抱歉，我没有找到相关的旅游信息。您可以尝试换一种问法，或者询问其他城市、景点的信息。"
                    print(f"💬 回答:\n{answer}\n")
                else:
                    print(f"📚 检索到 {len(search_results)} 条相关资料\n")
                    print("💬 回答:\n")

                    answer_stream = answer_generator.generate_stream(
                        query=user_input,
                        search_results=search_results,
                        intent=intent
                    )

                    answer = ""
                    for chunk in answer_stream:
                        print(chunk, end='', flush=True)
                        answer += chunk
                    print()

                history_store.add_user_message(session_id, user_input, {
                    'intent': intent,
                    'city': city,
                    'attraction': attraction
                })
                history_store.add_assistant_message(session_id, answer)
                history_store.save_history(session_id)

                session_manager.increment_message_count(session_id)

            except Exception as e:
                error_msg = f"抱歉，处理您的问题时出现了错误: {str(e)}"
                print(f"\n❌ {error_msg}\n")
                if logger:
                    logger.error(f"处理查询失败: {e}", exc_info=True)

        except KeyboardInterrupt:
            print("\n\n👋 检测到中断，再见！\n")
            logger.info("用户中断退出")
            break
        except EOFError:
            print("\n\n👋 再见！\n")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")
            logger.error(f"主循环错误: {e}", exc_info=True)


def main():
    """主函数"""
    global logger

    logger = setup_logging(log_level='INFO')
    logger.info("旅游知识库系统启动")

    # 检查命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == 'web':
            start_web()
            return
        elif mode == 'cli':
            print_welcome()
            start_cli()
            return
        elif mode == 'help':
            print_help()
            return
        else:
            print(f"❌ 未知模式: {mode}")
            print_help()
            return

    # 无参数时显示欢迎界面
    print_welcome()

    while True:
        try:
            choice = input("请选择启动模式 (web/cli/help): ").strip().lower()

            if choice in ['web', 'w']:
                start_web()
                break
            elif choice in ['cli', 'c']:
                start_cli()
                break
            elif choice in ['help', 'h', '?']:
                print_help()
            elif choice in ['quit', 'exit', 'q']:
                print("\n👋 再见！\n")
                break
            else:
                print("❌ 无效选择，请输入 web、cli 或 help\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见！\n")
            break
        except EOFError:
            print("\n\n👋 再见！\n")
            break


if __name__ == '__main__':
    main()
