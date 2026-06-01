from pathlib import Path
import sys
import uuid
import uvicorn

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.utils.path_util import PROJECT_ROOT
# 导入启动图对象
from app.query_process.agent.main_graph import query_graph


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

def run_query_graph(session_id: str, query: str, is_stream: bool):
    default_state = {"original_query": query, "session_id": session_id,"is_stream": is_stream}
    # 执行图流程
    try:
        query_graph.invoke(default_state)
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
    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        sse_queue = create_sse_queue(session_id)
    update_task_status(session_id, "processing",is_stream)

    if is_stream:
        # 立即返回流式响应，后台任务继续执行图流程并通过SSE推送结果
        background_tasks.add_task(run_query_graph, session_id, query_request.query,is_stream)
        print("结果正在生成中，请稍候...")
        return {
            "message": "结果正在生成中，请稍候...",
            "session_id": session_id
        }
    else:
        # 如果不使用流式响应，可以直接等待结果并返回（此处简化处理）
        run_query_graph(session_id, query_request.query, is_stream)
        answer = get_task_result(session_id,"answer","")
        return {
            "message": "处理完成",
            "session_id": session_id,
            "answer": answer,
            "done_list": get_done_task_list(session_id)
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

#查询当前会话历史记录
@app.get("/history/{session_id}")
async def get_history(session_id: str,limit: int = 50):
    """
    获取指定会话ID的历史记录
    """
    items = get_recent_messages(session_id, limit)  
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