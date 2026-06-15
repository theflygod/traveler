"""
LangGraph 工作流
定义旅游知识库的完整处理流程
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
import operator

from config.logging_config import get_logger
from intent.classifier import IntentClassifier
from retrieval.hybrid_search import HybridSearch
from generator.answer_generator import AnswerGenerator

logger = get_logger('workflow')


class TravelState(TypedDict):
    """旅游问答状态"""
    query: str
    session_id: str
    intent: str
    intent_confidence: float
    extracted_entities: Dict[str, str]
    search_results: List[Dict[str, Any]]
    answer: str
    error: str
    messages: Annotated[list, operator.add]


class TravelKnowledgeGraph:
    """旅游知识图谱工作流"""

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.hybrid_search = HybridSearch()
        self.answer_generator = AnswerGenerator()

        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

        logger.info("LangGraph工作流初始化完成")

    def _build_workflow(self) -> StateGraph:
        """构建工作流图"""
        workflow = StateGraph(TravelState)

        # 添加节点
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("search_knowledge", self._search_knowledge)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("handle_error", self._handle_error)

        # 设置入口点
        workflow.set_entry_point("classify_intent")

        # 添加边（简化版，不使用条件边）
        workflow.add_edge("classify_intent", "search_knowledge")
        workflow.add_edge("search_knowledge", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow

    def _classify_intent(self, state: TravelState) -> dict:
        """意图识别节点"""
        try:
            query = state['query']
            logger.info(f"意图识别: {query}")

            intent_result = self.intent_classifier.classify(query)

            return {
                'intent': intent_result.get('intent', '知识问答'),
                'intent_confidence': intent_result.get('confidence', 0.5),
                'extracted_entities': {
                    'city': intent_result.get('city', ''),
                    'attraction': intent_result.get('attraction', '')
                },
                'messages': [{'role': 'system', 'content': f'意图识别完成: {intent_result.get("intent")}'}]
            }

        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return {
                'error': f'意图识别失败: {str(e)}',
                'messages': [{'role': 'system', 'content': '意图识别出错'}]
            }

    def _search_knowledge(self, state: TravelState) -> dict:
        """知识检索节点"""
        try:
            query = state['query']
            entities = state.get('extracted_entities', {})

            logger.info(f"知识检索: query={query}")

            filters = {}
            if entities.get('city'):
                filters['city'] = entities['city']

            search_results = self.hybrid_search.search(
                query=query,
                top_k=5,
                filters=filters if filters else None
            )

            logger.info(f"检索到 {len(search_results)} 条结果")

            return {
                'search_results': search_results,
                'messages': [{'role': 'system', 'content': f'检索到{len(search_results)}条相关资料'}]
            }

        except Exception as e:
            logger.error(f"知识检索失败: {e}")
            return {
                'error': f'知识检索失败: {str(e)}',
                'search_results': [],
                'messages': [{'role': 'system', 'content': '检索出错'}]
            }

    def _generate_answer(self, state: TravelState) -> dict:
        """答案生成节点"""
        try:
            query = state['query']
            search_results = state.get('search_results', [])
            intent = state.get('intent', '知识问答')

            logger.info(f"生成答案: {len(search_results)}条资料")

            if not search_results:
                answer = "抱歉，我没有找到相关的旅游信息。您可以尝试换一种问法，或者询问其他城市、景点的信息。"
            else:
                answer = self.answer_generator.generate(
                    query=query,
                    search_results=search_results,
                    intent=intent
                )

            return {
                'answer': answer,
                'messages': [{'role': 'assistant', 'content': answer}]
            }

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return {
                'error': f'答案生成失败: {str(e)}',
                'answer': '抱歉，生成回答时出现错误。',
                'messages': [{'role': 'system', 'content': '答案生成出错'}]
            }

    def _handle_error(self, state: TravelState) -> dict:
        """错误处理节点"""
        error = state.get('error', '未知错误')
        logger.error(f"工作流错误: {error}")

        return {
            'answer': f'系统处理出现问题：{error}',
            'messages': [{'role': 'system', 'content': f'错误: {error}'}]
        }

    def run(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        运行工作流

        Args:
            query: 用户问题
            session_id: 会话ID

        Returns:
            完整的结果字典
        """
        initial_state = {
            'query': query,
            'session_id': session_id or 'default',
            'intent': '',
            'intent_confidence': 0.0,
            'extracted_entities': {},
            'search_results': [],
            'answer': '',
            'error': '',
            'messages': []
        }

        try:
            result = self.app.invoke(initial_state)

            logger.info(f"工作流执行完成: intent={result.get('intent')}, answer_len={len(result.get('answer', ''))}")

            return {
                'success': True,
                'query': query,
                'intent': result.get('intent', ''),
                'intent_confidence': result.get('intent_confidence', 0),
                'extracted_entities': result.get('extracted_entities', {}),
                'search_results': result.get('search_results', []),
                'answer': result.get('answer', ''),
                'message_count': len(result.get('messages', []))
            }

        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'answer': '系统错误，请稍后重试'
            }

    def run_stream(self, query: str, session_id: str = None):
        """
        流式运行工作流

        Args:
            query: 用户问题
            session_id: 会话ID

        Yields:
            每个节点的输出
        """
        initial_state = {
            'query': query,
            'session_id': session_id or 'default',
            'intent': '',
            'intent_confidence': 0.0,
            'extracted_entities': {},
            'search_results': [],
            'answer': '',
            'error': '',
            'messages': []
        }

        try:
            for event in self.app.stream(initial_state):
                yield event
        except Exception as e:
            logger.error(f"流式执行失败: {e}")
            yield {'error': str(e)}
