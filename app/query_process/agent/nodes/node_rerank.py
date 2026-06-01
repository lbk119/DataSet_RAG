import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[4]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from app.utils.task_utils import *
from dotenv import load_dotenv
from app.clients.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
load_dotenv()

# 动态 TopK 硬上限：最多取前 N 条（< 10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（> 1，且 < RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5

def step1_merge_docs(state) -> list:
    """
    阶段一：文档合并与标准化
    目标：将多路召回（本地知识库 + 联网搜索）的异构数据，统一合并为 Reranker 模型可处理的标准格式。
    输入来源：
    1. rrf_chunks (List[Dict]): 本地知识库检索结果（经 RRF 融合排序）。
    chunk 格式：{"id": ,"distance": ,"entity":{"chunk_id":,"content":,"item_name":}}

    2. web_search_docs (List[Dict]): 联网搜索结果（经 MCP 搜索返回）。
    [{"snippet": ,"title":,"url":},...]
    - 标准化文档列表，每项包含：
    - snippet: 用于重排序的核心文本
    - title: 标题（用于增强语义或展示）
    - url: 来源链接（本地为空，联网文档有）

    3.返回值格式：
    {
        "text": "内容",
        "chunk_id": rrf_chunks有 web_search_docs无,
        "title": "标题",
        "url": rrf_chunks无 web_search_docs有,
        "source": "local"(rrf) 或 "web"(mcp),
    }
    """

    # 1. 提取输入源
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    standardized_docs = []
    # 2. 标准化 RRF 文档
    for chunk in rrf_chunks:
        standardized_docs.append({
            "text": chunk["entity"]["content"],
            "chunk_id": chunk["entity"]["chunk_id"],
            "title": chunk["entity"].get("item_name", ""),
            "url": "",
            "source": "local"
        })

    # 3. 标准化网络文档
    for doc in web_search_docs:
        standardized_docs.append({
            "text": doc["snippet"],
            "chunk_id": "",
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "source": "web"
        })

    return standardized_docs

def step2_rerank_docs(state, docs) -> list:
    """
    阶段二：文档重排序
    目标：使用 Reranker 模型对合并后的文档进行打分排序，输出带有相关性得分的文档列表。
    输入：
    - docs (List[Dict]): 阶段一输出的标准化文档列表，每项包含 text、chunk_id、title、url、source。
    输出：
    - scored_docs (List[Dict]): 每个文档附加一个 "score" 字段，表示与用户查询的相关性得分。
    """
    # 1. 获取 Reranker 模型
    reranker = get_reranker_model()
    # 2. 获取用户查询问题
    query = state.get("rewritten_query", "")
    # 3. 获取文档文本并进行打分
    texts = [doc["text"] for doc in docs]
    query_texts = [(query, text) for text in texts]
    scored_docs = reranker.compute_score(query_texts, normalize=True)
    # 4. 将得分附加回文档列表
    for doc, score in zip(docs, scored_docs):
        doc["score"] = score
    docs.sort(key=lambda doc: doc.get("score", 0.0), reverse=True)
    return docs


def step3_dynamic_topk(scored_docs) -> list:
    """
    阶段三：动态 TopK（最多 10）
    基于 scored_docs（已按 score 降序排序）进行智能截断，
    核心逻辑：结合固定上下限+断崖阈值判断，避免机械取前N条，保留语义相关的连续文档集合
    :param scored_docs: 列表，元素为带score的文档字典，已按score降序排列，格式如
    [{"doc": 文档对象, "score": 相关性分数}, . ]
    :return: 列表，动态截断后的TopK文档列表，数量≤10
    """
    # 硬上限：最多取前10条，取全局常量与实际文档数的较小值（避免索引越界）
    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK # 硬下限：至少保留的文档数量（全局常量配置）
    gap_ratio = RERANK_GAP_RATIO # 相对断崖阈值：分数下降的相对比例阈值（全局常量配置）
    gap_abs = RERANK_GAP_ABS # 绝对断崖阈值：分数下降的绝对差值阈值（全局常量配置）
    if max_topk > min_topk:
        for i in range(min_topk - 1, max_topk - 1):
            prev_score = scored_docs[i].get("score", 0.0)
            curr_score = scored_docs[i + 1].get("score", 0.0)
            # 断崖判断：当前文档与前一文档的分数差距超过任一阈值（相对或绝对）
            if (prev_score - curr_score) / (abs(prev_score) + 1e-8) >= gap_ratio or (prev_score - curr_score) >= gap_abs:
                max_topk = i + 1 # 截断位置定在断崖前
                logger.info(f"动态TopK调整：原max_topk={max_topk}，因断崖调整为{max_topk}")
                break
    return scored_docs[:max_topk]

def node_rerank(state):
    """
    对rrf筛选后的文档和网络检索到的文档进行打分排序
    """
    function_name = sys._getframe().f_code.co_name
    add_running_task(state["session_id"], function_name, state.get("is_stream", False))
    logger.info(f"- {function_name} - 开始执行")
    # 阶段一：合并文档
    docs_to_rerank = step1_merge_docs(state)
    # 阶段二：对文档进行重排序
    scored_docs = step2_rerank_docs(state, docs_to_rerank)
    # 阶段三：动态 TopK（防断崖）
    reranked_docs = step3_dynamic_topk(scored_docs)
    state["reranked_docs"] = reranked_docs
    add_done_task(state["session_id"], function_name, state.get("is_stream", False))
    return {
        "reranked_docs": reranked_docs
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("> 启动 node_rerank 本地测试")
    print("="*50)
    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
    {"entity":{"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
    {"entity":{"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
    {"entity":{"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}} # 预期低分
    ]
    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
    {"title": "Rerank技术详解", "url": "http: web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
    {"title": "无关网页", "url": "http: web.com/2", "snippet": "今天天气不错，适合出去游玩"} # 预期低分
    ]
    mock_state = {
    "session_id": "test_rerank_session",
    "rewritten_query": "什么是RRF和Rerank？", # 查询意图：想了解这两个算法
    "rrf_chunks": mock_rrf_chunks,
    "web_search_docs": mock_web_docs,
    "is_stream": False
    }
    try:
    # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])
        print("\n" + "="*50)
        print("> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)
        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}. ")
        # 验证逻辑：
        # 预期 "local_1", "Rerank技术详解" 分数较高
        # 预期 "local_2","local_3", "无关网页" 分数较低，可能被截断或排在最后
        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")
        print("="*50)
    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")