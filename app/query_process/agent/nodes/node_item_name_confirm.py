import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages,save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client,create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
load_dotenv(find_dotenv())

def step2_extract_info(query: str, history_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    利用LLM从当前问题以及历史会话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）
    若商品名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
    :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
    :param history: 列表[字典] - 近期会话历史，每条消息含role/text等字段，格式：[{"role": "user/assistant", "text": "消息内容", "_id": "消息ID"}, ...]
    :return: 字典 - 提取结果，固定包含2个字段，格式：
    {
    "item_names": ["商品名1", "商品名2", . ], # 提取的商品名列表，无则空列表
    "rewritten_query": "改写后的完整问题" # 包含商品名的独立问题，无则返回原始query
    }
    """
    # 1. 初始化准备：获取LLM客户端，拼接历史会话为文本格式，加载并拼接提示词，构造LLM调用的消息列表
    llm_client = get_llm_client(json_mode=True)  # 获取LLM客户端，开启JSON输出模式
    history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history_messages])  # 将历史消息拼接为文本
    prompt_template = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)  # 加载提取信息的提示词模板
    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt_template)
    ]
    # 2. LLM调用与响应处理：调用LLM客户端获取响应，清理响应内容中的JSON代码块格式，解析为JSON字典
    try:
        response = llm_client.invoke(messages)
        response_content = response.content.strip()
        # 清理响应中的JSON代码块格式（如```json ... ```）
        if response_content.startswith("```json"):
            response_content = response_content.replace("```json", "").replace("```", "").strip()
        result = json.loads(response_content)  # 解析为JSON字典
    except Exception as e:
        logger.error(f"LLM响应解析失败: {e}")
        result = {"item_names": [], "rewritten_query": query}  # 解析失败则返回默认值
    # 3. 结果校验与异常处理：确保返回字典包含item_names/rewritten_query字段（缺失则补默认值），捕获所有异常并返回兜底结果
    if "item_names" not in result:
        result["item_names"] = []
    if "rewritten_query" not in result:
        result["rewritten_query"] = query
    return result

def step3_vectorize_and_query(item_names: List[str]) -> List[Dict[str, Any]]:
    """
    把分析出的item_names逐个向量化（BGEM3模型），并在Milvus向量数据库(kb_item_names)中执行混合搜索，获取匹配评分
    :param item_names: 字符串列表 - 提取到的商品名称列表，如 ["iPhone 14 Pro Max", "MacBook Air"]
    :return: 列表[字典] - 检索结果列表，每条结果包含id/score/metadata等字段，格式：
    [
        {
            "extracted_name": "提取的原始商品名", # 如"苹果15"
            "matches": [ # 该商品名的TopN匹配结果，无则空列表
                {
                    "item_name": "数据库中的商品名", # Milvus中存储的标准化商品名
                    "score": 0.98 # 混合搜索的相似度评分（0-1，越高越相似）
                },
                ...
            ]
        },
        ...
    ]
    """
    # 初始化最终返回结果列表，存储每个商品名的向量化查询结果
    results = []
    # 获取Milvus客户端连接
    milvus_client = get_milvus_client()
    if not milvus_client:
        logger.error("无法连接到Milvus数据库")
        return results  # 无法连接数据库则返回空结果
    collection_name = milvus_config.item_name_collection  # 从配置获取集合名称
    # 对所有商品名称批量生成BGEM3向量（稠密+稀疏），相比逐个生成提升处理效率
    embeddings = generate_embeddings(item_names) 
    for i in range(len(item_names)):
        item_result = {"extracted_name": item_names[i], "matches": []}  # 初始化当前商品名的结果结构
        try:
            # 构造当前商品名的混合搜索请求，包含稠密向量和稀疏向量部分
            search_request = create_hybrid_search_requests(
                dense_vector=embeddings["dense"][i],
                sparse_vector=embeddings["sparse"][i],
                limit=5  
            )
            # 执行混合搜索，获取原始结果列表（包含id/score/metadata等字段）
            search_results = hybrid_search(client=milvus_client, collection_name=collection_name, reqs=search_request,
                                           ranker_weights=(0.9, 0.1), # 稠/稀疏向量评分权重配比（和为1最佳）
                                           norm_score=True ,# 启用相似度评分归一化，确保不同商品名的评分在同一标准下可比较
                                           limit=5,output_fields=["item_name"]  # # 指定返回Milvus中存储的商品名字段
                                           )  
            # 解析搜索结果，提取标准化商品名和相似度评分，构建匹配结果列表
            if search_results and len(search_results) > 0:
                for res in search_results[0]:
                    # res格式：{"id": 数据库ID, "distance": 相似度评分, "entity":{"item_name": "标准化商品名"}}
                    match_info = {
                        "item_name": res.get("entity", {}).get("item_name"),  # 获取Milvus中存储的标准化商品名
                        "score": res.get("distance", 0)  # 获取相似度评分
                    }
                    item_result["matches"].append(match_info)  # 将匹配结果添加到当前商品名的结果中
        except Exception as e:
            logger.error(f"商品名 '{item_names[i]}' 的向量化查询失败: {e}")
        results.append(item_result)  # 将当前商品名的结果添加到最终返回列表中
    return results

def step4_align_item_names(query_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    对齐规则（优先级a>b>c>d）：
a 如果多条匹配结果评分超过0.85 → 优先取与原始提取名相同的，无则取分数最高的
c 如果无0.85分以上结果 → 取分数≥0.6的最高前5个作为候选
d 如果无0.6分及以上结果 → 不返回任何商品名（确认+候选均为空）
    :param query_results: 列表[字典] - 来自step3的原始查询结果列表，每条包含extracted_name和matches字段
    :return: 字典 - 最终确认的标准化商品名称列表和候选商品名称列表，如 {"confirmed_item_names": ["iPhone 14 Pro Max"], "options": ["MacBook Air"]}
    """
    confirmed_item_names = []
    optional_item_names = []
    for item in query_results:
# 提取原始的数据，商品名和匹配结果
        extracted_name = (item.get("extracted_name", "")).strip()
        # 获取匹配的商品名，无就获取空列表
        matches = item.get("matches", []) or []
        # 若无匹配结果，直接跳过当前商品名的对齐
        if not matches:
            continue
        # {
        # "item_name": , # 数据库标准化商品名
        # "score": , # 0-1相似度评分
        # }
        # 对匹配结果按评分* 降序* 排序（高分在前，优先取相似度高的）
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        high = [m for m in matches if m.get("score", 0) > 0.85]
        mid = [m for m in matches if m.get("score", 0) > 0.6]
        # 规则a: 多条高置信度结果（>0.85）
        if len(high) > 0:
            # 初始化选中结果为None，优先匹配原始提取名
            picked = None
            # 若原始提取名非空，优先取与原始名相同的匹配结果
            if extracted_name:
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        break
            # 如果没有与原始名相同的结果，则取分数最高的第一个结果
            if not picked:
                picked = high[0]
            # 将选中的结果加入确认商品名列表
            confirmed_item_names.append(picked.get("item_name"))
            continue # 跳过后续规则判断
        # 规则b: 无0.85分以上结果，取≥0.6分的最高前5个作为候选
        # 注：高置信度列表high为空时才会走到此处（规则a/b均不满足）
        if len(mid) > 0:
            for m in mid[:5]:
                optional_item_names.append(m.get("item_name"))
    return {
        "confirmed_item_names": list(set(confirmed_item_names)), # 去重，避免重复确认
        "options": list(set(optional_item_names)) # 去重，避免重复候选
    }

def step5_check_confirmation(state, aligned_results, history_messages):
    """
    根据对齐结果更新会话状态（State），决定后续流程分支
    1. 如果有确认的商品名，直接更新State的item_names并进入后续流程（如查询价格等）
    2. 如果无确认商品名但有候选商品名，更新State并生成一个新的问题，询问用户「你是想问xxx吗？」进行确认
    3. 如果既无确认商品名也无候选商品名，保持原问题不变，进入后续流程
    """
    confirmed_item_names = aligned_results.get("confirmed_item_names", [])
    optional_item_names = aligned_results.get("options", [])
    if confirmed_item_names:
        # 收集历史消息中未关联商品名的消息ID（需批量更新关联）
        ids_to_update = [msg["_id"] for msg in history_messages if not msg.get("item_names")]
        if ids_to_update:
            # 批量更新这些消息的item_names字段为已确认的商品名列表
            update_message_item_names(ids_to_update, confirmed_item_names)
        # 1. 有确认的商品名，直接更新State并进入后续流程
        state["item_names"] = confirmed_item_names
        logger.info(f"已确认商品名称: {confirmed_item_names}")
        # 若状态中存在旧答案，删除（避免干扰后续流程）
        if "answer" in state:
            del state["answer"]
        return state
    elif optional_item_names:
        # 2. 无确认商品名但有候选商品名，生成新的问题询问用户进行确认
        state["item_names"] = []
        options_text = ", ".join(optional_item_names[:3])
        answer = f"您是想问以下哪个产品：{options_text}？请明确一下型号。"
        state["answer"] = answer
        logger.info(f"未能确认商品名称，生成新问题: {answer}")
        return state
    else:
        # 3. 无确认也无候选
        state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
        state["item_names"] = []
        logger.info(f"未能提取到明确的商品名称，也没有候选项。")
    return state

def node_item_name_confirm(state):
    """
    Agent节点：确认商品名称
功能：从用户查询和历史会话中提取商品名称，进行向量化并在Milvus中查询匹配的标准化商品名称，返回提取的商品名称列表
输入：state字典，包含原始查询（original_query）和会话ID（session_id）等信息
输出：字典，包含提取的商品名称列表（item_names）和改写后的完整问题（rewritten_query）
流程：
1. 获取历史会话消息，保存当前用户查询
2. 调用LLM提取商品名称和改写问题
3. 对提取的商品名称进行向量化，并在Milvus中执行混合搜索，获取匹配结果
4. 根据Milvus搜索评分，逐个对齐step3提取的item_names，生成「确认商品名」和「候选商品名」
5. 根据对齐结果更新会话状态（State），决定后续流程分支
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state.get("is_stream", True)
    print(f"- {func_name}- 处理开始")
    add_running_task(session_id, func_name, is_stream)
    # 1. 获取历史记录,保存用户当前问题
    history_messages = get_recent_messages(session_id, limit=10)
    message_id = save_chat_message(session_id=session_id, role="user", text=state["original_query"],item_names=state.get("item_names", [])) 
    # 2. 调用LLM提取商品名称和改写问题
    extracted_info = step2_extract_info(state["original_query"], history_messages)
    item_names = extracted_info.get("item_names", [])
    rewritten_query = extracted_info.get("rewritten_query", state["original_query"])
    state["rewritten_query"] = rewritten_query
    if len(item_names) > 0:
        # 3. 对提取的商品名称进行向量化，并在Milvus中执行混合搜索，获取匹配结果
        query_results = step3_vectorize_and_query(item_names)
        # 4. 根据Milvus搜索评分，逐个对齐step3提取的item_names，生成「确认商品名」和「候选商品名」
        aligned_results = step4_align_item_names(query_results)
    else:
        logger.info(f"未提取到明确的商品名称")
    # 5.根据对齐结果更新会话状态（State），决定后续流程分支
    state = step5_check_confirmation(state, aligned_results, history_messages)
    # 6. 写入最终历史
    save_chat_message(session_id=session_id, role="user", text=state.get("original_query", ""),item_names=state.get("item_names", []))
    # 记录任务结束
    add_done_task(session_id, func_name, is_stream)
    logger.info(f"- {func_name}- 处理完成，确认商品名称: {state.get('item_names', [])}, 改写问题: {state.get('rewritten_query', '')}")
    return state


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
    "session_id": "test_session_001",
    "original_query": "HAK 180 烫金机怎么用？",
    "is_stream": False
    }
    print("> 开始测试 node_item_name_confirm. ")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)
        print("\n> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))
        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")
    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")