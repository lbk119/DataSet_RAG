import sys
import os

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

RRF_QA_CANDIDATE_LIMIT = int(os.getenv("RAG_RRF_QA_CANDIDATE_LIMIT", "30"))
RRF_EXAM_CANDIDATE_LIMIT = int(os.getenv("RAG_RRF_EXAM_CANDIDATE_LIMIT", "24"))


def get_chunk_id(chunk: Dict[str, Any]) -> Any:
    entity = chunk.get("entity", {})
    return (
        chunk.get("id")
        or chunk.get("chunk_id")
        or entity.get("chunk_id")
        or entity.get("id")
    )

def step2_reciprocal_rank(source_weights: List[tuple], k: int = 5) -> List[Dict[str, Any]]:
    """
        对embedding_chunks和hyde_embedding_chunks进行RRF融合排序
        chunk 格式：{"id": ,"distance": ,"entity":{"chunk_id":,"content":,"item_name":}}
    """
    score_dict = {} #chunk_id -> score
    chunks_dict = {} #chunk_id -> chunk
    for chunks, weight in source_weights:
        for rank, chunk in enumerate(chunks,start=1):
            chunk_id = chunk["id"] or chunk["entity"]["chunk_id"]
            score_dict[chunk_id] = (1.0 / (rank + 60)) * weight + score_dict.get(chunk_id, 0.0) # RRF得分计算公式
            chunks_dict.setdefault(chunk_id, chunk) # 存储chunk_id对应的实体信息
    # 根据得分排序并截取前K个结果
    # [(chunk,score),...]
    merged = []
    for chunk_id, score in score_dict.items():
        chunk = chunks_dict[chunk_id]
        merged.append((chunk, score))
    merged.sort(key=lambda x: x[1], reverse=True) # 按照得分降序排序
    merged = [item[0] for item in merged[:k]] # 取前K个chunk实体
    return merged


def _material_boost(chunk: Dict[str, Any], preferred_material_types: list[str]) -> float:
    entity = chunk.get("entity", {})
    material_type = entity.get("material_type", "")
    if material_type in preferred_material_types:
        index = preferred_material_types.index(material_type)
        return max(1.05, 1.35 - index * 0.1)
    return 1.0


def step2_reciprocal_rank_with_material_boost(source_weights: List[tuple], preferred_material_types: list[str], k: int = 5) -> List[Dict[str, Any]]:
    score_dict = {}
    chunks_dict = {}
    for chunks, weight in source_weights:
        for rank, chunk in enumerate(chunks, start=1):
            chunk_id = get_chunk_id(chunk)
            if not chunk_id:
                continue
            boost = _material_boost(chunk, preferred_material_types)
            score_dict[chunk_id] = (1.0 / (rank + 60)) * weight * boost + score_dict.get(chunk_id, 0.0)
            chunks_dict.setdefault(chunk_id, chunk)
    merged = [(chunks_dict[chunk_id], score) for chunk_id, score in score_dict.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in merged[:k]]

def node_rrf(state):
    """
    RRF (Reciprocal Rank Fusion) 倒数排名融合节点
    功能：
    将来自不同检索源（如 Embedding 检索、HyDE 检索、知识图谱检索等）的结果进行融合排序。
    RRF 是一种无需训练的算法，仅根据文档在不同列表中的排名来计算最终得分。
    步骤：
    1. 提取各路检索结果：从 state 中获取 embedding_chunks 和 hyde_embedding_chunks。
    2. 结果标准化：将不同格式的检索结果统一转换为包含 chunk_id 的实体列表。
    3. 设置权重：为不同来源分配权重（当前配置：Embedding=1.0, HyDE=1.0）。
    4. 执行 RRF：计算融合分数并重新排序。
    5. 结果截断：保留 Top K 个结果。
    6. 更新状态：将融合后的结果存入 state["rrf_chunks"]。
    """
    function_name = sys._getframe().f_code.co_name
    add_running_task(state["session_id"], function_name, state.get("is_stream", False))
    logger.info(f"- {function_name} - 开始执行")
    # 1. 提取各路检索结果
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])
    exam_chunks = state.get("exam_chunks", [])
    exam_semantic_chunks = state.get("exam_semantic_chunks", [])
    preferred_material_chunks = state.get("preferred_material_chunks", [])
    preferred_material_types = state.get("preferred_material_types", [])
    source_weights = [
        (exam_chunks, 4.0),  # 出卷模式下额外召回的试卷切片优先级最高
        (exam_semantic_chunks, 3.0),
        (preferred_material_chunks, 2.0),
        (embedding_chunks, 1.0),  # Embedding检索权重
        (hyde_embedding_chunks, 1.0)  # HyDE检索权重
    ]
    # 2. 应用带权重的RRF计算最终得分
    top_k = RRF_EXAM_CANDIDATE_LIMIT if state.get("mode") == "exam" else RRF_QA_CANDIDATE_LIMIT
    response = step2_reciprocal_rank_with_material_boost(source_weights, preferred_material_types, k=top_k)
    # 3. 更新状态并返回结果
    state["rrf_chunks"] = response
    add_done_task(state["session_id"], function_name, state.get("is_stream", False))
    logger.info(f"- {function_name} - 执行完成，融合后结果数量: {len(response)}")
    return {
        "rrf_chunks": response
    }

