from pathlib import Path
import sys
import uuid
import uvicorn
import threading
import shutil
from datetime import datetime
from typing import List

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from fastapi import FastAPI, BackgroundTasks, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.utils.path_util import PROJECT_ROOT
from app.clients.course_utils import course_response, ensure_course
from app.query_process.utils.attachment_utils import build_attachment_context
# 导入启动图对象
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_answer_output import node_answer_output


# 定义fastapi对象
app = FastAPI(title="询问系统", description="基于LangGraph的询问系统API")
# 配置跨域中间件，允许所有来源访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Startup warmup disabled: importing pymilvus in a background thread made the
# query service exit immediately in some Windows/uvicorn environments.
# The Milvus client is initialized lazily when the first query needs it.
# @app.on_event("startup")
def warmup_heavy_dependencies():
    def _warmup():
        try:
            print("[Warmup] start importing pymilvus in background...", flush=True)
            import pymilvus  # noqa: F401
            print("[Warmup] pymilvus import finished", flush=True)
        except Exception as e:
            print(f"[Warmup] pymilvus import failed: {e}", flush=True)

    threading.Thread(target=_warmup, daemon=True).start()

# 返回chat.html页面
@app.get("/chat.html", response_class=FileResponse)
async def get_chat_page():
    html_path = Path(__file__).parent.parent / "page" / "chat.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Chat page not found")
    return FileResponse(path=str(html_path), media_type="text/html")


def find_output_image(image_name: str) -> Path | None:
    output_dir = PROJECT_ROOT / "output"
    if not output_dir.exists():
        return None

    for image_path in output_dir.rglob(image_name):
        if image_path.is_file() and image_path.parent.name == "images":
            return image_path
    return None


@app.get("/images/{image_name}")
async def get_output_image(image_name: str):
    image_path = find_output_image(image_name)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path=str(image_path))

# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户的查询内容")
    session_id: str = Field(None, description="会话ID，默认为随机生成的UUID")
    is_stream: bool = Field(True, description="是否使用流式响应，默认为True")
    course_id: str = Field(None, description="课程ID")
    course_name: str = Field(None, description="课程名称")
    mode: str = Field("qa", description="qa/exam")

def run_query_graph(
        session_id: str,
        query: str,
        is_stream: bool,
        course_id: str = "",
        course_name: str = "",
        mode: str = "qa",
        attachment_context: str = "",
):
    print(f"[QueryGraph] start session={session_id}, course={course_name or course_id}, mode={mode}, query={query}", flush=True)
    default_state = {
        "original_query": query,
        "session_id": session_id,
        "is_stream": is_stream,
        "course_id": course_id,
        "course_name": course_name,
        "mode": mode or "qa",
        "attachment_context": attachment_context,
    }
    # 执行图流程
    try:
        print(f"[QueryGraph] node_item_name_confirm session={session_id}", flush=True)
        state = default_state
        state.update(node_item_name_confirm(state) or {})

        if not state.get("answer"):
            print(f"[QueryGraph] node_search_embedding session={session_id}", flush=True)
            state.update(node_search_embedding(state) or {})

            print(f"[QueryGraph] node_search_embedding_hyde session={session_id}", flush=True)
            state.update(node_search_embedding_hyde(state) or {})

            print(f"[QueryGraph] node_web_search_mcp session={session_id}", flush=True)
            state.update(node_web_search_mcp(state) or {})

            print(f"[QueryGraph] node_rrf session={session_id}", flush=True)
            state.update(node_rrf(state) or {})

            print(f"[QueryGraph] node_rerank session={session_id}", flush=True)
            state.update(node_rerank(state) or {})

        print(f"[QueryGraph] node_answer_output session={session_id}", flush=True)
        state.update(node_answer_output(state) or {})
        print(f"[QueryGraph] flow finished session={session_id}", flush=True)
        # 图流程执行完成后，更新任务状态和结果
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        update_task_status(session_id, "failed", is_stream)
        print(f"流程执行异常 {session_id}: {e}")
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)}) 
@app.post("/query")
async def query(query_request: QueryRequest, background_tasks: BackgroundTasks):
    # 生成或使用提供的session_id
    session_id = query_request.session_id or str(uuid.uuid4())
    is_stream = query_request.is_stream
    course = ensure_course(course_id=query_request.course_id, course_name=query_request.course_name)
    course_id = course["course_id"]
    course_name = course["course_name"]
    mode = query_request.mode or "qa"
    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        sse_queue = create_sse_queue(session_id)
    update_task_status(session_id, "processing",is_stream)

    if is_stream:
        # 立即返回流式响应，后台任务继续执行图流程并通过SSE推送结果。
        # 不使用 daemon thread，避免 CUDA/Milvus 等原生库在后台线程结束后触发进程异常退出。
        background_tasks.add_task(
            run_query_graph,
            session_id,
            query_request.query,
            is_stream,
            course_id,
            course_name,
            mode,
            "",
        )
        print("结果正在生成中，请稍候...")
        return {
            "message": "结果正在生成中，请稍候...",
            "session_id": session_id,
            "course": course
        }
    else:
        # 如果不使用流式响应，可以直接等待结果并返回（此处简化处理）
        run_query_graph(session_id, query_request.query, is_stream, course_id, course_name, mode, "")
        answer = get_task_result(session_id,"answer","")
        return {
            "message": "处理完成",
            "session_id": session_id,
            "course": course,
            "answer": answer,
            "done_list": get_done_task_list(session_id)
        }


def _save_query_attachments(session_id: str, files: List[UploadFile]) -> tuple[list[Path], Path]:
    date_dir = datetime.now().strftime("%Y%m%d")
    safe_session = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)
    work_dir = PROJECT_ROOT / "output" / "query_attachments" / date_dir / safe_session
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for upload in files or []:
        if not upload.filename:
            continue
        target = work_dir / Path(upload.filename).name
        with target.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        paths.append(target)
    return paths, work_dir


def run_query_graph_with_attachments(
        session_id: str,
        query: str,
        is_stream: bool,
        course_id: str,
        course_name: str,
        mode: str,
        saved_paths: list[Path],
        work_dir: Path,
):
    attachment_context = ""
    if saved_paths:
        try:
            print(f"[QueryGraph] analyzing attachments session={session_id}, count={len(saved_paths)}", flush=True)
            attachment_context = build_attachment_context(saved_paths, question=query, work_dir=work_dir)
        except Exception as e:
            print(f"[QueryGraph] attachment analysis failed session={session_id}: {e}", flush=True)
            attachment_context = f"Attachment analysis failed: {e}"
    run_query_graph(session_id, query, is_stream, course_id, course_name, mode, attachment_context)


@app.post("/query_with_files")
async def query_with_files(
        background_tasks: BackgroundTasks,
        query_text: str = Form(""),
        session_id: str = Form(None),
        is_stream: bool = Form(True),
        course_id: str = Form(None),
        course_name: str = Form(None),
        mode: str = Form("qa"),
        files: List[UploadFile] = File(default=[]),
):
    session_id = session_id or str(uuid.uuid4())
    course = ensure_course(course_id=course_id, course_name=course_name)
    course_id = course["course_id"]
    course_name = course["course_name"]
    saved_paths, work_dir = _save_query_attachments(session_id, files)
    final_query = query_text.strip() or "请根据我上传的附件进行分析并回答。"
    if is_stream:
        create_sse_queue(session_id)
    update_task_status(session_id, "processing", is_stream)

    if is_stream:
        background_tasks.add_task(
            run_query_graph_with_attachments,
            session_id,
            final_query,
            is_stream,
            course_id,
            course_name,
            mode or "qa",
            saved_paths,
            work_dir,
        )
        return {
            "message": "结果正在生成中，请稍候...",
            "session_id": session_id,
            "course": course,
            "attachments": [p.name for p in saved_paths],
        }

    attachment_context = ""
    if saved_paths:
        attachment_context = build_attachment_context(saved_paths, question=final_query, work_dir=work_dir)
    run_query_graph(session_id, final_query, is_stream, course_id, course_name, mode or "qa", attachment_context)
    answer = get_task_result(session_id, "answer", "")
    return {
        "message": "处理完成",
        "session_id": session_id,
        "course": course,
        "answer": answer,
        "done_list": get_done_task_list(session_id),
        "attachments": [p.name for p in saved_paths],
    }
    
@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    return StreamingResponse(sse_generator(session_id, request), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive","X-Accel-Buffering": "no"})

# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}


@app.get("/courses")
async def list_courses_api():
    return course_response()


@app.post("/courses")
async def create_course_api(course_name: str = Form(...)):
    course = ensure_course(course_name=course_name)
    return {"code": 200, "course": course}

#查询当前会话历史记录
@app.get("/history/{session_id}")
async def get_history(session_id: str,limit: int = 50, course_id: str = ""):
    """
    获取指定会话ID的历史记录
    """
    items = get_recent_messages(session_id, limit, course_id=course_id)  
    if items is None:
        raise HTTPException(status_code=404, detail="History not found")
    return {"session_id": session_id, "items": items}

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """
    删除指定会话ID的历史记录
    """
    deleted_count = clear_history(session_id)
    return {"session_id": session_id, "deleted_count": deleted_count}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
