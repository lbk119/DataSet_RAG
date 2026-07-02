import sys
import os

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.utils.task_utils import add_running_task,add_done_task
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
from app.query_process.agent.state import QueryGraphState
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string
load_dotenv(find_dotenv())

EXAM_CONTEXT_LIMIT = 30
INITIAL_RECALL_LIMIT = int(os.getenv("RAG_INITIAL_RECALL_LIMIT", "30"))
OUTPUT_FIELDS = [
    "chunk_id",
    "content",
    "item_name",
    "course_id",
    "course_name",
    "material_type",
    "file_title",
    "title",
    "parent_title",
    "part",
    "exam_year",
    "exam_question_no",
    "exam_question_title",
    "exam_question_type",
    "exam_score",
    "exam_topics",
    "is_reference_answer",
    "topics",
    "primary_topic",
]


def infer_query_intent(query: str, mode: str = "qa") -> str:
    text = query or ""
    if mode == "exam" or any(word in text for word in ["出卷", "模拟卷", "期末试卷", "预测", "往年试卷", "题型结构"]):
        return "exam"
    if any(word in text for word in ["已知", "求", "计算", "证明", "构造", "解", "迭代", "公式为", "步骤"]):
        return "problem"
    return "concept"


def preferred_material_types(intent: str) -> list[str]:
    if intent == "exam":
        return ["exam", "exam_answer"]
    if intent == "problem":
        return ["homework", "exam_answer", "exam"]
    return ["textbook", "slides", "courseware", "homework"]


def _build_course_expr(course_id: str, mode: str = "qa") -> str:
    if not course_id:
        return ""
    expr = f'course_id == "{escape_milvus_string(course_id)}"'
    if mode == "exam":
        # 考试预测优先试卷，但保留课件/教材作为补充由排序阶段决定。
        return expr
    return expr

def _fetch_exam_chunks(client, collection_name: str, course_id: str, limit: int = EXAM_CONTEXT_LIMIT):
    if not course_id:
        return []
    expr = f'course_id == "{escape_milvus_string(course_id)}" and material_type == "exam"'
    try:
        rows = client.query(
            collection_name=collection_name,
            filter=expr,
            output_fields=OUTPUT_FIELDS,
            limit=limit,
        )
    except TypeError:
        rows = client.query(
            collection_name=collection_name,
            filter=expr,
            output_fields=OUTPUT_FIELDS,
        )
        rows = rows[:limit]
    except Exception as e:
        logger.warning(f"出卷模式试卷切片查询失败，将只使用向量召回结果: {e}")
        return []

    chunks = []
    for index, row in enumerate(rows or []):
        chunk_id = row.get("chunk_id") or row.get("id") or f"exam-{index}"
        chunks.append({
            "id": chunk_id,
            "distance": 1.0,
            "entity": row,
        })
    logger.info(f"出卷模式额外召回往年试卷切片数量: {len(chunks)}")
    return chunks

def node_search_embedding(state:QueryGraphState)->QueryGraphState:
    """
    核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
    流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
    :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
    {
    "session_id": str, # 会话唯一标识
    "rewritten_query": str, # step3改写后的完整用户问题（含商品名）
    "item_names": list[str], # step6已确认的标准化商品名列表
    "is_stream": bool/None # 是否为流式响应，可选
    }
    :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
    {
    "embedding_chunks": List[Dict] # Milvus检索结果列表，无结果则为空列表
    }
    """
    function_name = sys._getframe().f_code.co_name
    add_running_task(state["session_id"], function_name, state.get("is_stream", False))
    query = state.get("rewritten_query", "")
    item_names = state.get("item_names", [])
    course_id = state.get("course_id", "")
    query_intent = infer_query_intent(query or state.get("original_query", ""), state.get("mode", "qa"))
    state["query_intent"] = query_intent
    state["preferred_material_types"] = preferred_material_types(query_intent)
    if not item_names and not course_id:
        logger.warning(f"- {function_name} - 没有确认的商品名称，无法执行基于商品过滤的检索")
        state["embedding_chunks"] = []
        add_done_task(state["session_id"], function_name, state.get("is_stream", False))
        return state
    logger.info(f"- {function_name} - 开始执行，输入状态: rewritten_query='{query}', item_names={item_names}")
    collection_name = milvus_config.chunks_collection
    client = get_milvus_client()
    exam_chunks = []
    if state.get("mode") == "exam" and course_id:
        print(f"[node_search_embedding] exam mode: fetching exam chunks before embedding session={state['session_id']}", flush=True)
        exam_chunks = _fetch_exam_chunks(client, collection_name, course_id)
        if exam_chunks:
            state["exam_chunks"] = exam_chunks

    # 1. 对改写后的用户问题执行向量化，生成BGEM3稠密+稀疏向量
    print(f"[node_search_embedding] importing embedding utils session={state['session_id']}", flush=True)
    from app.lm.embedding_utils import generate_embeddings
    print(f"[node_search_embedding] generating query embedding session={state['session_id']}", flush=True)
    embeddings = generate_embeddings([query])
    dense_vector = embeddings.get("dense")[0]  # 获取第一个文本的稠密向量
    sparse_vector = embeddings.get("sparse")[0]  # 获取第一个文本的稀疏向量
    # 2. 准备Milvus向量数据库连接相关配置，指定检索的集合
    # 3. 构造带商品名过滤的混合搜索请求
    if course_id:
        expr = _build_course_expr(course_id, state.get("mode", "qa"))
    else:
        expr_str = ", ".join([f'"{name}"' for name in item_names])  # 构造过滤表达式字符串
        expr = f'item_name in [{expr_str}]'  # 构造最终过滤表达式
    search_requests = create_hybrid_search_requests(dense_vector, sparse_vector, expr=expr, limit=INITIAL_RECALL_LIMIT)
    # 4. 执行Milvus稠密+稀疏混合向量检索
    search_results = hybrid_search(
        client,
        collection_name,
        search_requests,
        (0.9, 0.1),
        True,
        INITIAL_RECALL_LIMIT,
        output_fields=OUTPUT_FIELDS,
    ) 
    state["embedding_chunks"] = search_results[0] if search_results else []  # 获取第一个请求的检索结果，若无结果则为空列表
    preferred_chunks = []
    if course_id and state.get("preferred_material_types"):
        material_values = ", ".join([f'"{escape_milvus_string(item)}"' for item in state["preferred_material_types"]])
        preferred_expr = f'course_id == "{escape_milvus_string(course_id)}" and material_type in [{material_values}]'
        preferred_requests = create_hybrid_search_requests(dense_vector, sparse_vector, expr=preferred_expr, limit=INITIAL_RECALL_LIMIT)
        preferred_results = hybrid_search(
            client,
            collection_name,
            preferred_requests,
            (0.9, 0.1),
            True,
            INITIAL_RECALL_LIMIT,
            output_fields=OUTPUT_FIELDS,
        )
        preferred_chunks = preferred_results[0] if preferred_results else []
        state["preferred_material_chunks"] = preferred_chunks
    exam_semantic_chunks = []
    if state.get("mode") == "exam" and course_id:
        exam_expr = f'course_id == "{escape_milvus_string(course_id)}" and material_type == "exam"'
        exam_requests = create_hybrid_search_requests(dense_vector, sparse_vector, expr=exam_expr, limit=16)
        exam_results = hybrid_search(
            client,
            collection_name,
            exam_requests,
            (0.9, 0.1),
            True,
            16,
            output_fields=OUTPUT_FIELDS,
        )
        exam_semantic_chunks = exam_results[0] if exam_results else []
        state["exam_semantic_chunks"] = exam_semantic_chunks
        state["exam_chunks"] = exam_chunks
    add_done_task(state["session_id"], function_name, state.get("is_stream", False))
    return {
        "embedding_chunks": state["embedding_chunks"],
        "preferred_material_chunks": preferred_chunks,
        "exam_chunks": exam_chunks,
        "exam_semantic_chunks": exam_semantic_chunks,
    }


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
    "session_id": "test_search_embedding_001",
    "rewritten_query": "HAK 180 烫金机使用说明", # 模拟改写后的查询
    "item_names": ["HAK180烫金机"], # 模拟已确认的商品名
    "is_stream": False
    }
    print("\n> 开始测试 node_search_embedding 节点. ")
    try:
    # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n> 测试完成！检索到 {len(chunks)} 条结果")
        if chunks:
            print("\n> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}. ")
        else:
            print("\n> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")
    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)
