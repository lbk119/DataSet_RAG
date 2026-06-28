import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import uvicorn

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
add_running_task,
add_done_task,
get_done_task_list,
get_running_task_list,
update_task_status,
get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import import_graph # LangGraph全流程编译实例
from app.core.logger import logger # 项目统一日志工具
from app.conf.minio_config import minio_config # MinIO配置对象实例
from app.clients.course_utils import course_response, ensure_course

app = FastAPI(title="知识库导入服务", description="基于LangGraph的知识库导入API", version="1.0.0")
# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
    allow_credentials=True,  # 允许携带凭证（如Cookies）
)

@app.get("/import.html",response_class=FileResponse)
async def get_import_page():
    """
    提供一个简单的HTML页面，允许用户上传文件并触发导入流程。
    这个页面可以通过浏览器访问，作为测试和演示接口使用。
    """
    html_path = os.path.join(PROJECT_ROOT, "app", "import_process", "page", "import.html")
    if not os.path.exists(html_path):
        logger.error(f"导入页面文件不存在: {html_path}")
        raise HTTPException(status_code=404, detail="Import page not found")
    return FileResponse(path=html_path, media_type="text/html")

# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
def run_graph_task(
        task_id: str,
        local_dir: str,
        local_file_path: str,
        course_id: str = "",
        course_name: str = "",
        material_type: str = "other",
):
    """
    后台任务函数：执行LangGraph全流程。
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看
    1. 初始化状态：构造ImportGraphState实例，设置输入文件路径等初始参数。
    2. 执行流程：调用import_graph.run()，传入初始状态，触发整个流程的执行。
    3. 错误处理：捕获流程执行中的异常，更新任务状态为失败，并记录错误信息。
    4. 成功完成：流程执行成功后，更新任务状态为完成。
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing") # 更新任务状态为处理中
        logger.info(f"任务 {task_id} 开始执行，输入文件: {local_file_path}")
        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        state = get_default_state() # 获取默认状态实例
        state["task_id"] = task_id # 注入任务ID
        state["course_id"] = course_id
        state["course_name"] = course_name
        state["material_type"] = material_type
        state["local_dir"] = local_dir # 注入工作目录（中间文件输出目录）
        state["local_file_path"] = local_file_path # 注入输入文件路径
        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        for event in import_graph.stream(state):
            for node_name, _ in event.items():
                logger.info(f"任务 {task_id} - 节点 {node_name} 执行完成")
                # 实时更新任务的已完成节点列表（done_list），供前端轮询查看
                add_done_task(task_id, node_name) # 将完成的节点加入done_list
        # 4. 全流程执行完成，更新任务状态为完成
        update_task_status(task_id, "completed") # 更新任务状态为完成
        logger.info(f"任务 {task_id} 执行完成")
    except Exception as e:
        # 4. 错误处理
        update_task_status(task_id, "failed") # 更新任务状态为失败
        logger.error(f"任务 {task_id} 执行失败，错误信息: {str(e)}")

@app.get("/courses")
async def list_courses_api():
    return course_response()


@app.post("/courses")
async def create_course_api(course_name: str = Form(...)):
    course = ensure_course(course_name=course_name)
    return {"code": 200, "course": course}


@app.post("/upload",summary="上传文件并触发导入流程",description="接收用户上传的多文件，保存到服务器，并触发后台任务执行LangGraph全流程。")
async def upload_file(
        files: List[UploadFile] = File(...),
        background_tasks: BackgroundTasks = None,
        course_id: str = Form(None),
        course_name: str = Form(None),
        material_type: str = Form("other"),
):
    """
    文件上传核心接口
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 将文件上传至MinIO对象存储，做持久化保存
    4. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    5. 实时更新任务状态，供前端轮询监控进度
    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    date_str = datetime.now().strftime("%Y%m%d")
    course = ensure_course(course_id=course_id, course_name=course_name)
    course_id = course["course_id"]
    course_name = course["course_name"]
    material_type = material_type or "other"
    date_dir = os.path.join(PROJECT_ROOT, "output", "courses", course_id, date_str)
    os.makedirs(date_dir, exist_ok=True)
    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    task_ids = [] # 存储所有生成的任务ID，供前端展示
    for upload_file in files:
        # 生成唯一TaskID：使用UUID，确保每个文件对应一个独立的处理任务
        task_id = str(uuid.uuid4())
        task_ids.append(task_id) # 将生成的TaskID加入列表
        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file") 
        # 4. 构建本地文件路径：output/YYYYMMDD/task_id_filename.ext（按任务ID分层，避免冲突）
        local_dir = os.path.join(date_dir, task_id)
        os.makedirs(local_dir, exist_ok=True)
        local_file_path = os.path.join(local_dir, upload_file.filename)
        # 5. 将上传的文件保存到本地临时目录（后续MinIO上传/文件解析均基于此文件）
        with open(local_file_path, "wb") as f:
            shutil.copyfileobj(upload_file.file, f)
        logger.info(f"文件 {upload_file.filename} 已保存到 {local_file_path}")
        # 6. 上传文件到MinIO对象存储（持久化保存，便于后续访问）
        try:
            minio_client = get_minio_client() # 获取MinIO客户端实例
            bucket_name = minio_config.bucket_name # 从配置获取桶名称
            object_name = f"courses/{course_id}/{date_str}/{upload_file.filename}" # 构建对象名称，按课程/日期分层
            minio_client.fput_object(bucket_name=bucket_name, object_name=object_name, file_path=local_file_path,content_type=upload_file.content_type)
            logger.info(f"文件 {upload_file.filename} 已上传到MinIO，桶: {bucket_name}, 对象: {object_name}")
        except Exception as e:
            logger.error(f"文件 {upload_file.filename} 上传到MinIO失败，错误信息: {str(e)}")
        # 7. 启动后台任务执行LangGraph全流程，传入必要参数（TaskID、文件路径等）
        add_done_task(task_id, "upload_file") # 标记「文件上传」阶段完成
        background_tasks.add_task(run_graph_task, task_id, local_dir, local_file_path, course_id, course_name, material_type) # 添加后台任务
        logger.info(f"任务 {task_id} 已启动后台处理，文件路径: {local_file_path}")
    # 8. 返回响应：包含所有生成的TaskID，供前端展示和后续查询使用
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {"code": 200,"message": f"Files uploaded successfully, total: {len(files)}","task_ids": task_ids, "course": course}

@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO
    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
    "code": 200,
    "task_id": task_id,
    "status": get_task_status(task_id), # 任务全局状态：pending/processing/completed/failed
    "done_list": get_done_task_list(task_id), # 已完成的节点/阶段列表
    "running_list": get_running_task_list(task_id) # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
