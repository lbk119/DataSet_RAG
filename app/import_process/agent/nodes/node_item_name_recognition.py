# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
import os
import sys
from typing import List, Dict, Any, Tuple

if __package__ in (None, ""):
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage
# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task,add_done_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt
from app.utils.escape_milvus_string_utils import escape_milvus_string
# - 配置参数 (Configuration) -
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

def step1_extract_input(state: ImportGraphState) -> Tuple[str, List[Dict[str, Any]]]:
    """
    步骤1：提取输入数据
    从状态对象中提取文件标题和切片列表，进行基本校验
    :param state: ImportGraphState对象，包含整个流程的状态信息
    :return: 文件标题（str）和切片列表（List[Dict]）
    """
    file_title = state.get("file_title", "").strip()
    chunks = state.get("chunks", [])
    if not file_title:
        logger.warning("步骤1 - 文件标题为空，可能影响后续识别准确性")
    if not chunks:
        logger.warning("步骤1 - 切片列表为空，无法进行商品名称识别")
    return file_title, chunks

def step2_build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    步骤2：构建大模型输入上下文
    基于切片内容构建识别上下文，包含前K个切片的内容摘要
    核心作用：
    1. 限制切片数量：仅取前k个切片，避免上下文过长,DEFAULT_ITEM_NAME_CHUNK_K
    2. 限制字符长度：单切片+总上下文双重字符限制，适配大模型输入上限,SINGLE_CHUNK_CONTENT_MAX_LEN、CONTEXT_TOTAL_MAX_CHARS
    3. 格式化内容：带序号的结构化格式，提升大模型识别精度
    4. 过滤无效切片：跳过空内容/非字典类型切片，保证上下文有效性
    参数说明：
    chunks: 文本切片列表（每个元素为字典，需包含"title"和"content"键）
    :return: 构建好的上下文字符串，供大模型识别使用
    """
    if not chunks:
        logger.warning("步骤2 - 输入切片列表为空，无法构建上下文")
        return ""
    context_parts = []
    total_chars = 0
    for i, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        content = chunk.get("content", "").strip()
        title = chunk.get("title", "").strip()
        if content:
            truncated_content = content[:SINGLE_CHUNK_CONTENT_MAX_LEN]  # 截断单切片内容
            piece = f"切片{i} - 切片标题：{title}\n切片内容：{truncated_content}"
            context_parts.append(piece)
            total_chars += len(piece)
            if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
                logger.info("步骤2 - 上下文字符数达到上限，停止添加更多切片")
                break
    context = "\n\n".join(context_parts)
    #二次截断
    if len(context) > CONTEXT_TOTAL_MAX_CHARS:
        context = context[:CONTEXT_TOTAL_MAX_CHARS]
    return context
def step3_llm_recognition(file_title: str, context: str) -> str:
    """
    步骤 3: 调用大模型实现商品名称/型号精准识别
    核心逻辑：
    1. 上下文为空 → 直接返回file_title（兜底，无需调用大模型）
    2. 上下文非空 → 加载标准化prompt模板，构建大模型对话消息
    3. 调用大模型后对返回结果做清洗，过滤无效字符
    4. 大模型返回空/调用异常 → 均返回file_title兜底，保证流程不中断
    参数：
    file_title: 处理后的文件标题（异常/空值时的兜底值）
    context: 步骤2构建的结构化切片上下文（大模型识别的核心依据）
    返回值：
    str: 清洗后的商品名称（异常/空值时返回原始file_title）
    """
    if not context:
        logger.warning("步骤3 - 上下文为空，无法调用大模型识别，直接返回文件标题作为商品名称")
        return file_title
    try:
        human_prompt = load_prompt("item_name_recognition",file_title=file_title,context=context)
        system_prompt = load_prompt("product_recognition_system")
        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=human_prompt)
        messages = [system_message, human_message]
        llm_client = get_llm_client(json_mode=False)  # 获取大模型客户端实例，json_mode=False表示返回原始文本
        response = llm_client.invoke(messages)
        # 兼容不同LLM客户端的返回格式，提取文本内容
        if isinstance(response, list) and response and hasattr(response[0], "content"):
            raw_item_name = response[0].content.strip()
        elif hasattr(response, "content"):
            raw_item_name = response.content.strip()
        else:
            logger.warning("步骤3 - 大模型返回格式不兼容，无法提取内容，使用文件标题兜底")
            return file_title
        cleaned_item_name = raw_item_name.replace(" ", "").replace("\n", "").replace("\r","").replace("\t", "")  # 清洗大模型返回结果，去除空格和控制字符
        if not cleaned_item_name:
            logger.warning("步骤3 - 大模型识别结果为空或无效，使用文件标题兜底")
            return file_title
        return cleaned_item_name
    except Exception as e:
        logger.error(f"步骤3 - 大模型识别异常，错误信息：{str(e)}，使用文件标题兜底", exc_info=True)
        return file_title
    
def step4_backfill_data(state: ImportGraphState, chunks: List[Dict[str, Any]], item_name: str):
    """
    步骤4：回填数据
    将识别到的商品名称回填到状态中的每个切片，便于后续使用
    核心逻辑：
    1. 遍历切片列表，将item_name字段添加到每个切片字典中
    2. 更新状态对象中的chunks列表，保持数据一致性
    参数说明：
    state: ImportGraphState对象，整个流程的状态载体
    chunks: 原始切片列表（每个元素为字典）
    item_name: 步骤3识别到的商品名称字符串
    返回值：无（直接修改输入状态对象）
    """
    state["item_name"] = item_name  # 在状态对象中添加item_name字段，便于全局访问
    for chunk in chunks:
        chunk["item_name"] = item_name  # 在每个切片字典中添加item_name字段
    state["chunks"] = chunks  # 更新状态对象中的chunks列表

def step5_generate_vectors(item_name: str) -> Tuple[List[float], List[Dict[int, float]]]:
    """
    步骤5：为item_name生成向量
    基于识别的商品名称生成稠密向量和稀疏向量
    核心逻辑：
    1. 调用BGE-M3模型生成稠密向量，获取固定长度的浮点数列表
    2. 调用稀疏向量生成方法，获取稀疏向量（字典形式，key为维度索引，value为权重值）
    3. 对稀疏向量进行归一化处理，保证向量长度为1，提升后续检索准确性
    参数说明：
    item_name: 步骤3识别到的商品名称字符串，是生成向量的核心输入
    返回值：
    Tuple[List[float], List[Dict[int, float]]]: 稠密向量（浮点数列表）和归一化后的稀疏向量（字典列表）
    """
    if not item_name:
        logger.warning("步骤5 - 输入的商品名称为空，无法生成有效向量，返回空列表")
        return [], []
    vector_result = generate_embeddings([item_name])  # 调用BGE-M3模型生成稠密+稀疏混合向量嵌入
    dense_vector = vector_result["dense"][0]  # 提取第一个文本的稠密向量（列表形式）
    sparse_vector = vector_result["sparse"][0]  # 提取第一个文本的稀疏向量（字典形式）
    return dense_vector, sparse_vector

def step6_store_in_milvus(state: ImportGraphState, file_title: str, item_name: str, dense_vector: List[float], sparse_vector: Dict[int, float]):
    """
    步骤6：将切片数据和生成的向量存入Milvus数据库
    核心逻辑：
        1. 客户端获取：获取单例Milvus客户端，连接失败则跳过
        2. 集合初始化：无集合则创建（定义Schema+索引），有集合则直接使用（保留原有配置）
        3. 幂等性处理：删除同名商品数据，避免重复存储
        4. 数据插入：构造符合Schema的数据，非空向量才添加
        5. 集合加载：插入后强制加载集合，确保数据立即可查/Attu可见
    参数说明：
    state: ImportGraphState对象，整个流程的状态载体
    file_title: 文件标题字符串，作为记录的一部分存储
    item_name: 步骤3识别到的商品名称字符串，作为记录的一部分存储
    dense_vector: 步骤5生成的稠密向量列表，用于Milvus的向量字段
    sparse_vector: 步骤5生成的稀疏向量字典，用于Milvus的向量字段（需适配存储格式）
    返回值：无（直接将数据插入Milvus数据库）
    """
    milvus_client = get_milvus_client()  # 获取Milvus客户端实例
    if not milvus_client:
        logger.error("步骤6 - Milvus客户端未初始化，无法存储数据")
        return
    collection_name = milvus_config.item_name_collection
    if not collection_name:
        logger.error("步骤6 - 未配置ITEM_NAME_COLLECTION，无法存储商品名称向量")
        return
    if not milvus_client.has_collection(collection_name=collection_name):
        logger.info(f"步骤6 - Milvus集合[{collection_name}]不存在，开始创建")
        # 定义集合Schema，包含基本字段和向量字段（稠密+稀疏）
        schema = milvus_client.create_schema(auto_id=True,enable_dynamic_field=True)  # 开启自增主键，启用动态字段

        # 3.2. Add fields to schema
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True,auto_id=True) # 添加自增主键字段：INT64类型，唯一标识每条数据
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=len(dense_vector))
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
        milvus_client.create_collection(collection_name=collection_name, schema=schema,index_params=index_params)  # 创建集合并应用Schema
        logger.info(f"步骤6 - Milvus集合[{collection_name}]创建成功")
    # 幂等性处理：删除同名商品数据，避免重复存储（核心：先加载集合才能删除）
    milvus_client.load_collection(collection_name=collection_name)  # 加载集合，确保数据可见
    milvus_client.delete(collection_name=collection_name, filter=f"item_name == '{escape_milvus_string(item_name)}'")  # 删除同名商品数据，保证幂等性
    # 构造待插入数据，非空向量才添加
    insert_data = {
        "file_title": file_title,
        "item_name": item_name,
    }
    if dense_vector:
        insert_data["dense_vector"] = dense_vector
    if sparse_vector:
        insert_data["sparse_vector"] = sparse_vector
    milvus_client.insert(collection_name=collection_name, data=[insert_data])  # 插入数据
    milvus_client.load_collection(collection_name=collection_name)  # 插入
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    【核心节点】商品主体名称识别（node_item_name_recognition）
    整体流程：提取输入→构建上下文→大模型识别→回填数据→生成向量→存入Milvus
    核心目的：利用大模型从文档切片中精准识别商品/主体名称，并生成双路向量（稠密+稀疏）存入数据库
    后续扩展点：支持多主体识别、增加商品属性提取、对接其他向量库等
    :param state: 项目状态字典（ImportGraphState），必须包含chunks/file_title等关键信息
    :return: 更新后的状态字典，新增item_name键，且chunks列表中每个元素新增item_name字段
    """
    function_name = sys._getframe().f_code.co_name  # 获取当前函数名，便于日志记录
    logger.info(f"开始执行{function_name}")
    add_running_task(state["task_id"], function_name)  # 记录当前任务状态，便于监控和调度
    try:
        # 1. 提取输入数据：从状态中获取切片列表和文件标题
        file_title,chunks = step1_extract_input(state)
        if not chunks:
            logger.warning(f"{function_name} - 输入切片列表为空，无法进行商品名称识别")
            raise ValueError("输入切片列表为空")
        # 2. 构建大模型输入上下文：基于切片内容和文件标题构建识别上下文
        context = step2_build_context(chunks)
        # 3. 大模型识别：调用大模型接口，传入构建的上下文，获取识别结果
        item_name = step3_llm_recognition(file_title, context)
        # 4. 回填数据：将识别到的商品名称回填到状态中的每个切片，便于后续使用
        step4_backfill_data(state, chunks, item_name)
        # 5. 生成向量：基于识别的商品名称生成稠密向量和稀疏向量,输出：dense_vector（List[float]）、sparse_vector（Dict[int, float]）
        dense_vector, sparse_vector = step5_generate_vectors(item_name)
        # 6. 存入Milvus：将切片数据和生成的向量存入Milvus数据库，便于后续检索和使用
        step6_store_in_milvus(state, file_title, item_name, dense_vector, sparse_vector)
        add_done_task(state["task_id"], function_name)  # 记录当前任务完成状态，便于监控和调度
        logger.info(f"{function_name}执行成功，识别到的商品名称：{item_name}")
    except Exception as e:
        logger.error(f"{function_name}执行失败，错误信息：{str(e)}", exc_info=True)
        # 失败时不修改状态，直接返回原状态，保证流程健壮性
    return state



# ===================== 本地测试方法（直接运行调试，无需启动LangGraph）=====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
    1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
    2. 确保大模型、Milvus、BGE-M3服务均可正常访问
    3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
    直接运行该函数：if __name__ = "__main__":
    test_node_item_name_recognition()
    """
    logger.info("= 开始执行商品名称识别节点本地测试 = ")
    try:
    # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
        "task_id": "test_task_123456", # 测试任务ID
        "file_title": "华为Mate60 Pro手机使用说明书", # 模拟文件标题
        "file_name": "华为Mate60Pro说明书.pdf", # 模拟原始文件名（兜底用）
        # 模拟文本切片列表（上游切片节点产出，含title/content字段）
        "chunks": [
            {
            "title": "产品简介",
            "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
            },
            {
            "title": "拍照功能",
            "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
            },
            {
            "title": "电池参数",
            "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
            }
        ]
        })
        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)
        # 3. 打印测试结果（调试用）
        logger.info("= 商品名称识别节点本地测试完成 = ")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")
        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name and milvus_client.has_collection(collection_name=collection_name):
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            res = milvus_client.query(
            collection_name=collection_name,
            filter=f'item_name == "{escape_milvus_string(item_name)}"',
            output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")
        else:
            logger.warning("Milvus客户端、集合名不可用或集合尚未创建，跳过本地测试中的Milvus检索校验")
    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)
# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()