import sys
import os

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
from app.query_process.agent.state import QueryGraphState
from app.conf.milvus_config import milvus_config
load_dotenv(find_dotenv())

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
    if not item_names:
        logger.warning(f"- {function_name} - 没有确认的商品名称，无法执行基于商品过滤的检索")
        state["embedding_chunks"] = []
        add_done_task(state["session_id"], function_name, state.get("is_stream", False))
        return state
    logger.info(f"- {function_name} - 开始执行，输入状态: rewritten_query='{query}', item_names={item_names}")
    # 1. 对改写后的用户问题执行向量化，生成BGEM3稠密+稀疏向量
    embeddings = generate_embeddings([query])
    dense_vector = embeddings.get("dense")[0]  # 获取第一个文本的稠密向量
    sparse_vector = embeddings.get("sparse")[0]  # 获取第一个文本的稀疏向量
    # 2. 准备Milvus向量数据库连接相关配置，指定检索的集合
    collection_name = milvus_config.chunks_collection
    # 3. 构造带商品名过滤的混合搜索请求
    expr_str = ", ".join([f'"{name}"' for name in item_names])  # 构造过滤表达式字符串
    expr = f'item_name in [{expr_str}]'  # 构造最终过滤表达式
    search_requests = create_hybrid_search_requests(dense_vector, sparse_vector, expr = expr, limit = 10)
    # 4. 执行Milvus稠密+稀疏混合向量检索
    client = get_milvus_client()
    search_results = hybrid_search(client, collection_name, search_requests,(0.9, 0.1),True,5,output_fields=["chunk_id", "content", "item_name"]) 
    state["embedding_chunks"] = search_results[0] if search_results else []  # 获取第一个请求的检索结果，若无结果则为空列表
    add_done_task(state["session_id"], function_name, state.get("is_stream", False))
    return {"embedding_chunks": state["embedding_chunks"]}


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