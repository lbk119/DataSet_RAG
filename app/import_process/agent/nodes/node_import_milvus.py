import os
import sys
from pathlib import Path
from typing import List, Dict, Any

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

def step2_prepare_milvus_collection():
    """
    Milvus准备步骤2：连接客户端并确保集合存在
    核心逻辑：
    1. 客户端连接：使用配置文件中的参数连接Milvus服务器，获取客户端实例
    2. 集合检查：检查切片集合是否存在，不存在则创建，确保Schema正确
    3. 索引创建：为向量字段创建索引，提升后续查询性能
    返回：
    milvus_client - 已连接的Milvus客户端实例，供后续操作使用
    异常处理：
    连接失败或集合准备失败抛出异常，由调用方捕获处理
    """
    milvus_client = get_milvus_client()
    if milvus_client is None:
        raise ConnectionError("Milvus客户端未初始化，请检查MILVUS_URL配置和Milvus服务状态")
    if not milvus_client.has_collection(CHUNKS_COLLECTION_NAME):
        logger.info(f"步骤6 - Milvus集合[{CHUNKS_COLLECTION_NAME}]不存在，开始创建")
        # 定义集合Schema，包含基本字段和向量字段（稠密+稀疏）
        schema = milvus_client.create_schema(auto_id=True,enable_dynamic_field=True)  # 开启自增主键，启用动态字段

        # 3.2. Add fields to schema
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True,auto_id=True) # 添加自增主键字段：INT64类型，唯一标识每条数据
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", # Name of the vector field to be indexed
            index_type="HNSW", # Type of the index to create
            index_name="dense_vector_index", # Name of the index to create
            metric_type="COSINE", # Metric type used to measure similarity
            # M: 图中每个节点的最大连接数(常用16-64)
            # efConstruction: 构建索引时的搜索范围(越大建索引越慢，但精度越高，常用100-200)
            # 不同数据体量的推荐建议(万级)：
            # 10000 条数据：M=16, efConstruction=200
            # 50000 条数据：M=32, efConstruction=300
            # 100000 条数据：M=64, efConstruction=400
            params={"M": 16, "efConstruction": 200}
        )
        index_params.add_index(
            field_name="sparse_vector", # Name of the vector field to be indexed
            # 稀疏倒排索引 专门为稀疏向量（比如文本的 TF-IDF 向量、关键词权重向量，特点是大部分元素为 0，只有少数维度有值）设计的倒排索引，是稀疏向量检索的标配索引类型。
            index_type="SPARSE_INVERTED_INDEX", # Type of the index to create
            index_name="sparse_vector_index", # Name of the index to create
            metric_type="IP", # Metric type used to measure similarity
            params={"M": 16, "efConstruction": 200}
        )
        milvus_client.create_collection(collection_name=CHUNKS_COLLECTION_NAME, schema=schema,index_params=index_params)  # 创建集合并应用Schema
        logger.info(f"步骤2 - Milvus集合[{CHUNKS_COLLECTION_NAME}]创建成功")
    return milvus_client

def step3_clean_old_data(milvus_client, chunks):
    """
    Milvus准备步骤3：幂等清理旧数据，避免重复存储
    核心逻辑：
    1. 提取切片中的item_name列表，作为删除条件
    2. 构建删除表达式，批量删除Milvus中同item_name的旧数据
    3. 确保删除操作幂等执行，不存在则跳过，避免误删
    参数：
    milvus_client - 已连接的Milvus客户端实例
    chunks - 当前批次待入库的切片列表，包含item_name字段
    异常处理：
    删除失败抛出异常，由调用方捕获处理
    """
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)  # 加载集合，确保数据可见
    item_names = list(set(chunk.get("item_name") for chunk in chunks if chunk.get("item_name")))
    if not item_names:
        logger.warning("步骤3 - 没有有效的item_name可用于幂等清理，跳过删除")
        return
    delete_expression = " OR ".join([f'item_name == "{name}"' for name in item_names])
    logger.info(f"步骤3 - 开始幂等清理Milvus中item_name在{item_names}的旧数据")
    milvus_client.delete(collection_name=CHUNKS_COLLECTION_NAME, filter=delete_expression)

def step4_insert_chunks(milvus_client, chunks) -> List[Dict[str, Any]]:
    """
    步骤4：批量插入切片数据到Milvus+主键回填
    核心逻辑：
    1. 移除手动chunk_id：因auto_id=True，Milvus自动生成主键，避免冲突
    2. 批量插入数据：提升入库效率，减少Milvus连接次数
    3. 回填chunk_id：将Milvus生成的自增主键回填到切片，供下游业务使用
    参数：
    milvus_client - MilvusClient实例
    chunks: List[Dict[str, Any]] - 待入库的切片列表
    返回：
    List[Dict[str, Any]] - 回填了chunk_id的切片列表
    """
    for chunk in chunks:
        chunk.pop("chunk_id", None)  # 移除原有chunk_id，避免与Milvus自增主键冲突
    insert_result = milvus_client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=chunks)
    generated_ids = insert_result.get("ids", [])  # 获取Milvus返回的自增主键列表
    if len(generated_ids) != len(chunks):
        raise ValueError(f"插入后返回的主键数量{len(generated_ids)}与插入切片数量{len(chunks)}不匹配")
    for chunk, chunk_id in zip(chunks, generated_ids):
        chunk["chunk_id"] = str(chunk_id)  # 回填Milvus生成的chunk_id到切片
    return chunks
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph核心节点：Milvus切片数据入库主流程
    执行流程（串行执行，一步一校验，保证数据一致性）：
    1. 输入校验：验证切片有效性、向量字段完整性，提取向量维度
    2. 环境准备：连接Milvus，集合不存在则自动创建Schema+索引
    3. 幂等清理：删除同item_name旧数据，避免重复存储
    4. 批量插入：预处理数据后批量入库，回填Milvus自增chunk_id
    5. 状态更新：将回填了chunk_id的切片更新回全局状态，供下游使用
    参数：
    state: ImportGraphState - 流程全局状态对象，包含chunks、task_id等数据
    返回：
    ImportGraphState - 更新后的状态对象，chunks字段回填chunk_id
    异常处理：
    任一步骤失败抛出ValueError，终止节点执行，保证数据不脏写
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f"开始执行{function_name}")
    task_id = state.get("task_id", "")
    add_running_task(task_id, function_name)
    try:
        # 步骤1：输入数据有效性校验
        chunks = state.get("chunks", [])
        if not chunks:
            raise ValueError("输入切片列表为空，无法入库")
        # 步骤2：Milvus客户端连接+集合准备（自动建表）
        milvus_client = step2_prepare_milvus_collection()
        # 步骤3：幂等清理旧数据（根据item_name删除）
        step3_clean_old_data(milvus_client, chunks)
        # 步骤4：批量预处理数据并插入Milvus，回填chunk_id
        updated_chunks = step4_insert_chunks(milvus_client, chunks)
        # 步骤5：更新全局状态，回填chunk_id供下游使用
        state["chunks"] = updated_chunks
        add_done_task(task_id, function_name)  # 记录当前任务完成状态，便于监控和调度
        logger.info(f"{function_name}执行完成，成功入库{len(updated_chunks)}条切片")
    except Exception as e:
        logger.error(f"{function_name}执行失败: {e}", exc_info=True)
        raise ValueError(f"{function_name}执行失败: {e}")
    return state


if __name__ == '__main__':
    # - 单元测试 -
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv
    # 加载环境变量 (自动寻找项目根目录的 .env)
    project_root = Path(__file__).resolve().parents[4]
    load_dotenv(project_root / ".env")
    # 构造测试数据
    dim = 1024
    test_state = {
    "task_id": "test_milvus_task",
    "chunks": [
    {
    "content": "Milvus 测试文本 1",
    "title": "测试标题",
    "item_name": "测试项目_Milvus", # 必须有 item_name，用于幂等清理
    "parent_title":"test.pdf",
    "part":1,
    "file_title": "test.pdf",
    "dense_vector": [0.1] * dim, # 模拟 Dense Vector
    "sparse_vector": {1: 0.5, 10: 0.8} # 模拟 Sparse Vector
    }
    ]
    }
    print("正在执行 Milvus 导入节点测试. ")
    try:
    # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
             # 执行节点函数
            result_state = node_import_milvus(test_state)
    # 验证结果
        chunks = result_state.get("chunks", [])
        if chunks and chunks[0].get("chunk_id"):
            print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
        else:
            print("❌ 测试失败：未能获取 chunk_id")
    except Exception as e:
        print(f"❌ 测试失败: {e}")