import sys
import os

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.utils.task_utils import add_running_task, add_done_task
from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string
load_dotenv(find_dotenv())

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

def step1_generate_hyde_document(query)->str:
    """
    Step 1: 生成假设文档
    输入：用户问题（query）
    输出：HyDE文档（hyde_document）
    1. 加载HyDE专用Prompt模板。
    2. 将用户问题填入Prompt，调用LLM生成假设文档。
    3. 返回生成的HyDE文档。
    """
    if not query:
        logger.warning("输入查询为空，无法生成HyDE文档")
        raise ValueError("输入查询不能为空")
    try:
        logger.info(f"Step 1: 生成HyDE文档 - 输入查询: '{query}'")
        prompt_template = load_prompt("hyde_prompt",rewritten_query=query)
        from app.lm.lm_utils import get_llm_client
        llm_client = get_llm_client()
        response = llm_client.invoke(prompt_template)
        hyde_document=response.content.strip()
        logger.info(f"Step 1: 生成HyDE文档 - 生成的HyDE文档长度: {len(hyde_document)}")
        return hyde_document
    except Exception as e:
        logger.error(f"Step 1: 生成HyDE文档失败: {e}")
        raise e
    
def step2_search_with_hyde(query, hyde_document, item_names, req_limit = 10,top_k=5,
                           ranker_weight=(0.9,0.1),norm_score: bool = True,output_fields=["chunk_id", "content", "item_name"], course_id: str = ""):
    """
    阶段2：利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片。
    :param query: 改写后的查询
    :param  hyde_document: Step 1 生成的假设性文档
    :param item_names: 商品名称列表，用于元数据过滤 (item_name in [. ])
    :param req_limit: Milvus 搜索时的候选召回数量
    :param top_k: 最终返回的 Top K 结果数量
    :param ranker_weights: 混合检索权重 (Dense, Sparse)
    :param norm_score: 是否对分数进行归一化
    :param output_fields: 返回结果中包含的字段
    :return: 检索结果列表
    """
    if not query or not hyde_document:
        logger.warning("输入查询或HyDE文档为空，无法执行混合检索")
        raise ValueError("输入查询和HyDE文档不能为空")
    # 1. 拼接查询与假设文档，形成更丰富的语义上下文
    combined_text = f"{query}\n{hyde_document}"
    # 2. 生成向量 (Dense + Sparse)
    from app.lm.embedding_utils import generate_embeddings
    embedding = generate_embeddings([combined_text])
    # 3. 准备 Milvus 检索的过滤条件，基于商品名称进行元数据过滤
    collection_name = milvus_config.chunks_collection
    if course_id:
        expr = f'course_id == "{escape_milvus_string(course_id)}"'
    else:
        expr_str = ", ".join([f'"{item}"' for item in item_names])
        expr = f'item_name in [{expr_str}]' if item_names else ""
    try:
        reqs = create_hybrid_search_requests(embedding["dense"][0], embedding["sparse"][0], expr=expr, limit=req_limit)
        client = get_milvus_client()
        response = hybrid_search(client, collection_name, reqs, ranker_weights=ranker_weight, norm_score=norm_score, limit=top_k, output_fields=output_fields)
        logger.info(f"Step 2: 混合检索 - 检索到的结果数量: {len(response)}")
        return response
    except Exception as e:
        logger.error(f"Step 2: 混合检索失败: {e}")
        raise e
def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
    2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
    3. 混合检索 (Step 2)：
    - 将“用户问题 + 假设文档”合并，生成BGE-M3稠密+稀疏向量。
    - 在Milvus中执行混合检索（带商品名过滤），召回最相似的知识切片。
    4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态state的hyde_embedding_chunks字段。
    """
    function_name = sys._getframe().f_code.co_name
    add_running_task(state["session_id"], function_name, state.get("is_stream", False))
    # 1. 参数提取
    query = state.get("rewritten_query", "")
    if not query:
        query = state.get("original_query", "")
    item_names = state.get("item_names", [])
    course_id = state.get("course_id", "")
    if state.get("mode") == "exam" and state.get("exam_chunks"):
        logger.info(f"- {function_name} - 出卷模式已获取试卷切片，跳过HyDE扩展检索")
        add_done_task(state["session_id"], function_name, state.get("is_stream", False))
        return {
            "hyde_embedding_chunks": []
        }
    logger.info(f"- {function_name} - 开始执行，输入状态: rewritten_query='{query}', item_names={item_names}")
    hyde_document = ""
    try:
        # 2. 生成假设文档
        hyde_document = step1_generate_hyde_document(query)
    except Exception as e:
        logger.error(f"- {function_name} - 生成HyDE文档失败: {e}")
        return {}
    
    try:
        # 3. 混合检索
        res = step2_search_with_hyde(
            query,
            hyde_document,
            item_names,
            top_k=8 if state.get("mode") == "exam" else 5,
            output_fields=OUTPUT_FIELDS,
            course_id=course_id,
        )
        add_done_task(state["session_id"], function_name, state.get("is_stream", False))
        return {
            "hyde_embedding_chunks": res[0] if res else []
        }
    except Exception as e:
        logger.error(f"- {function_name} - 混合检索失败: {e}")
        return {}
    

if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "="*50)
    print("> 启动 node_search_embedding_hyde 本地测试")
    print("="*50)
    # 模拟输入状态
    mock_state = {
    "session_id": "test_hyde_session_001",
    "original_query": "HAK180烫金机怎么操作？",
    "rewritten_query": "HAKK180烫金机的具体操作步骤是什么？",
    "item_names": ["HAK180烫金机"],
    "is_stream": False
    }
    try:
    # 运行节点
        result = node_search_embedding_hyde(mock_state)
        print("\n" + "="*50)
        print("> 测试结果摘要:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")
        if chunks:
            print(f"Top Chunk Score: {chunks[0].get('distance')}")
        print("="*50)
    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
