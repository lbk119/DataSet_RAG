import sys
import os
from pathlib import Path
from typing import Any, List, Dict

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger
def step3_generate_embeddings(model, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    向量化核心步骤3：批量生成稠密/稀疏双向量
    核心逻辑（分批执行，每批独立异常处理）：
    1. 文本拼接：item_name（商品名）+ 换行 + content（切片内容），强化核心特征
    2. 批量调用：传入拼接后的文本，生成批量双向量
    3. 向量绑定：为每个切片复制原数据，新增dense_vector/sparse_vector字段
    4. 异常兜底：单批次失败则保留原切片数据，继续处理下一批次
    参数：
    chunks: List[Dict[str, Any]] - 校验通过的文本切片列表，含item_name/content字段
    model: Any - 步骤2初始化的BGE-M3模型实例
    返回：
    List[Dict[str, Any]] - 带向量字段的文本切片列表，异常批次保留原数据
    关键配置：
    batch_size: 每批处理5条，可根据服务器显存大小调整（显存大则调大，反之调小）
    """
    batch_size = 5
    output_chunks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            # 文本拼接：item_name + , + content
            texts = [f"商品：{chunk.get('item_name','')},介绍：{chunk.get('content','')}" for chunk in batch]
            # 批量生成双向量
            vectors = generate_embeddings(texts)
            # 向量绑定：为每个切片新增dense_vector/sparse_vector字段
            for chunk, dense_vec, sparse_vec in zip(batch, vectors['dense'], vectors['sparse']):
                new_chunk = chunk.copy()  # 复制原数据，避免修改原对象
                new_chunk['dense_vector'] = dense_vec
                new_chunk['sparse_vector'] = sparse_vec
                output_chunks.append(new_chunk)
        except Exception as e:
            logger.error(f"批次{i//batch_size}向量化失败: {e}", exc_info=True)
            output_chunks.extend(batch)  # 失败批次保留原数据继续处理下一批次
    return output_chunks

def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph核心节点：BGE-M3文本向量化处理
    主流程（串行执行，全流程异常隔离）：
    1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
    2. 模型初始化：获取BGE-M3单例模型实例，避免重复加载
    3. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
    4. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用
    参数：
    state: ImportGraphState - 流程全局状态对象，包含上游传入的chunks等数据
    返回：
    ImportGraphState - 更新后的状态对象，chunks字段新增dense_vector/sparse_vector
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"开始执行{function_name}")
    task_id = state.get("task_id", "")
    add_running_task(task_id, function_name)
    try:
        # 1. 输入校验
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValueError("输入chunks无效，必须为非空列表")
        # 2. 模型初始化
        model = get_bge_m3_ef()
        # 3. 批量向量化
        output_chunks = step3_generate_embeddings(model, chunks)
        
        # 4. 状态更新
        state["chunks"] = output_chunks
        add_done_task(task_id, function_name)
        logger.info(f"{function_name} - 执行完成，已生成向量")
    except Exception as e:
        logger.error(f"{function_name} - 执行异常: {e}")
    return state



# ==========================================
# 本地单元测试入口
# 功能：独立验证向量化节点全链路逻辑，无需启动整个LangGraph流程
# 适用场景：本地开发、调试、模型有效性验证
# ==========================================
from dotenv import load_dotenv
if __name__ == '__main__':
    # 加载环境变量：定位项目根目录下的.env，读取模型路径/设备等配置
    project_root = Path(__file__).resolve().parents[4]
    load_dotenv(project_root / ".env")
    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
    "task_id": "test_task_embedding_001", # 测试任务ID
    "chunks": [ # 模拟带item_name的文本切片（上游商品名称识别节点产出）
    {
    "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
    "title": "测试文档标题",
    "item_name": "测试项目",
    "file_title": "测试文件.pdf"
    },
    {
    "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
    "title": "测试文档标题2",
    "item_name": "测试项目",
    "file_title": "测试文件.pdf"
    }
    ]
    })
    # 执行本地测试
    logger.info("= BGE-M3向量化节点本地单元测试启动 = ")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])
        # 打印测试结果统计
        logger.info(f"= 向量化节点本地测试完成 = ")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")
        # 验证向量生成结果（打印向量字段是否存在）
        for idx, chunk in enumerate(result_chunks):
            has_dense = "dense_vector" in chunk
            has_sparse = "sparse_vector" in chunk
            logger.info(f"第{idx + 1}条切片：稠密向量生成{'' if has_dense else '未'}成功 | 稀疏向量生成{'' if has_sparse else '未'}成功")   
    except Exception as e:
        logger.error(f"= 向量化节点本地测试失败 = " f"错误原因：{str(e)}",exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")