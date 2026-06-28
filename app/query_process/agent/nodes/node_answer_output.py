import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[4]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.clients.mongo_history_utils import save_chat_message
import re
_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000
MAX_EXAM_CONTEXT_CHARS = 30000

MATH_ENV_PATTERN = re.compile(
    r"^\s*\\begin\{(cases|aligned|align\*?|array|matrix|pmatrix|bmatrix|vmatrix|equation\*?|split|gather\*?)\}"
)

def _strip_line_prefix(line: str) -> str:
    return re.sub(r"^(?:>\s*|[-*]\s+|\d+\.\s+|#+\s+)", "", line.strip()).strip()

def _is_likely_math_line(line: str) -> bool:
    cleaned = _strip_line_prefix(line)
    if not cleaned or len(cleaned) > 700:
        return False
    if MATH_ENV_PATTERN.match(cleaned):
        return True
    if re.match(r"^\\(boxed|frac|omega|sum|int|lim|sqrt|left|right)", cleaned):
        return True

    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    has_math_command = re.search(
        r"\\(frac|Rightarrow|rightarrow|begin|end|boxed|quad|text|cdot|leq|geq|neq|omega|xi|in|sum|int|sqrt|left|right)",
        cleaned,
    )
    has_math_symbol = re.search(r"[=^_{}]|[+\-*/×÷]|≤|≥|≠|≈", cleaned)
    starts_like_math = re.match(r"^(\(?\d+\)?\s*)?([A-Za-z]\\?|'|\\[A-Za-z]+|\(|\[|{|[0-9.-])", cleaned)
    return bool(has_math_symbol and ((starts_like_math and chinese_count <= 4) or (has_math_command and chinese_count <= 10)))

def normalize_answer_markdown(answer: str) -> str:
    """Make model math output easier for the chat page to render cleanly."""
    if not answer:
        return answer

    lines = answer.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    in_code = False
    in_display_math = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue

        if in_code:
            out.append(line)
            i += 1
            continue

        if stripped == "$$":
            in_display_math = not in_display_math
            out.append(line)
            i += 1
            continue

        if in_display_math or stripped.startswith("$$") or stripped.endswith("$$"):
            out.append(line)
            i += 1
            continue

        env_match = MATH_ENV_PATTERN.match(_strip_line_prefix(line))
        if env_match:
            env = env_match.group(1)
            block = [_strip_line_prefix(line)]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if f"\\end{{{env}}}" in lines[i]:
                    break
                i += 1
            out.extend(["$$", "\n".join(block).strip(), "$$"])
            i += 1
            continue

        if _is_likely_math_line(line):
            out.extend(["$$", _strip_line_prefix(line), "$$"])
        else:
            out.append(line)
        i += 1

    return "\n".join(out)

def step1_check_existing_answer(state: QueryGraphState) -> bool:
    """
    判断state中是否已有答案
    :param state: QueryGraphState对象
    :return: 如果已有答案返回True，否则返回False
    """
    answer = state.get("answer", "")
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id", "")
    if answer:
        logger.info("已有答案，直接输出")
        if is_stream:
            # 流式输出答案
            push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
        else:
            # 非流式输出，直接返回完整答案
            set_task_result(session_id, "answer", answer)
        return True
    return False
    
def step2_load_prompt(state: QueryGraphState) -> str:
    """
    根据state中的问题、重写问题、历史对话、提问商品（item_names）、 重排内容 组织prompt
    :param state: QueryGraphState对象
    :return: 组织好的prompt字符串
    """
    rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
    history = state.get("history", [])
    item_names = state.get("item_names", [])
    reranked_docs = state.get("reranked_docs", [])
    course_name = state.get("course_name", "")
    mode = state.get("mode", "qa")
    attachment_context = (state.get("attachment_context", "") or "").strip()
    # 处理reranked_docs
    docs = []
    exam_docs = []
    support_docs = []
    used_length = 0
    max_context_chars = MAX_EXAM_CONTEXT_CHARS if mode == "exam" else MAX_CONTEXT_CHARS
    for i,doc in enumerate(reranked_docs,start=1):
        title = doc.get("title", "")
        text = doc.get("text", "")
        source = doc.get("source", "")
        material_type = doc.get("material_type", "")
        score = doc.get("score", 0.0)
        content = f"Document {i} (Source: {source}, Type: {material_type}, Title: {title}, Score: {score}): {text}\n"
        if used_length + len(content) > max_context_chars:
            logger.info(f"文档内容超过最大限制，已截断。当前已使用字符数: {used_length}, 新文档字符数: {len(content)}, 最大限制: {max_context_chars}")
            break
        if mode == "exam" and material_type == "exam":
            exam_docs.append(content)
        elif mode == "exam":
            support_docs.append(content)
        else:
            docs.append(content)
        used_length += len(content)
    if mode == "exam":
        context = (
            "【往年试卷结构依据】\n"
            + ("\n".join(exam_docs) if exam_docs else "未检索到 material_type=exam 的往年试卷切片，请降低结构确定性并说明依据不足。")
            + "\n\n【补充课程资料】\n"
            + ("\n".join(support_docs) if support_docs else "无补充课程资料。")
        )
    else:
        context = "\n".join(docs)
    if attachment_context:
        context = (
            "【用户本次上传附件解析结果】\n"
            + attachment_context
            + "\n\n【课程知识库检索结果】\n"
            + (context or "无")
        )
    # 处理history
    history_str = ""
    if history and len(history) > 0:
        for i, turn in enumerate(history, start=1):
            role = turn.get("role", "")
            text = turn.get("text", "")
            current_turn = ""
            if role == "user" and text:
                current_turn = f"Turn {i}:\nUser: {text}\n"
            elif role == "assistant" and text:
                current_turn = f"Turn {i}:\nAssistant: {text}\n"
            history_str += current_turn
            used_length += len(current_turn)
            if used_length >= max_context_chars:
                logger.info(f"历史对话内容超过最大限制，已截断。当前已使用字符数: {used_length}, 新历史对话字符数: {len(current_turn)}, 最大限制: {max_context_chars}")
                break
    else:
        history_str = "没有历史对话记录"
        logger.info("没有历史对话记录")
    # 处理item_names
    item_names_str = ", ".join(item_names)

    # 加载prompt模板
    if mode == "exam":
        prompt_template = load_prompt("exam_generation", context=context, history=history_str, course_name=course_name, question=rewritten_query)
    else:
        prompt_template = load_prompt("answer_out", context=context, history=history_str, item_names=item_names_str or course_name, question=rewritten_query)
    return prompt_template

def step3_generate_answer(state: QueryGraphState, prompt_template: str) -> str:
    """
    调用大模型生成答案
    :param state: QueryGraphState对象
    :param prompt_template: 组织好的prompt字符串
    :return: 大模型生成的答案字符串
    """
    from app.lm.lm_utils import get_llm_client

    llm_client = get_llm_client()
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id", "")
    answer = ""
    if is_stream:
        delta_buffer = []
        buffer_chars = 0

        def flush_delta_buffer():
            nonlocal delta_buffer, buffer_chars
            if not delta_buffer:
                return
            push_to_session(session_id, SSEEvent.DELTA, {"delta": "".join(delta_buffer)})
            delta_buffer = []
            buffer_chars = 0

        for chunk in llm_client.stream(prompt_template):
            delta = chunk.content
            answer += delta
            if delta:
                delta_buffer.append(delta)
                buffer_chars += len(delta)
                if buffer_chars >= 120 or delta.endswith(("\n", "。", "！", "？", ".", "!", "?")):
                    flush_delta_buffer()
        flush_delta_buffer()
        answer = normalize_answer_markdown(answer)
        set_task_result(session_id, "answer", answer)
    else:
        response = llm_client.invoke(prompt_template)
        answer = normalize_answer_markdown(response.content)
        set_task_result(session_id, "answer", answer)
    state["answer"] = answer
    return answer

def step4_extract_image_urls(state: QueryGraphState) -> list:
    """
    辅助方法：从文档列表中提取图片URL
    核心逻辑：
    1. 遍历所有相关文档（包括本地知识库切片和联网搜索结果）。
        local的chunk中的text字段 {text: "![](url)}->url
        网络mcp中的{url:""}

    2. 策略一：直接检查文档的 'url' 字段（常见于联网搜索结果）。
    - 验证后缀名是否为图片格式 (.jpg, .png 等)。
    3. 策略二：使用正则表达式扫描文档 'text' 正文内容（常见于本地 Markdown 文档）。
    - 匹配 Markdown 图片语法: ![alt text](image_url)。
    4. 对提取到的 URL 进行去重处理，返回唯一图片列表。
    :param docs: 文档列表，每个文档为字典格式
    :return: 图片 URL 字符串列表
    """
    images = []
    image_set = set()
    # 定义图片格式正则表达式
    image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    reranked_docs = state.get("reranked_docs", [])
    # 1. 遍历文档列表
    for doc in reranked_docs:
        # 策略一：检查 'url' 字段
        url = doc.get("url", "")
        if url and url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg')) and url not in image_set:
            image_set.add(url)
            images.append(url)
        # 策略二：扫描 'text' 字段
        text = doc.get("text", "")
        markdown_image_urls = image_pattern.findall(text)
        for img_url in markdown_image_urls:
                if img_url not in image_set:
                    image_set.add(img_url)
                    images.append(img_url)
    logger.info(f"提取到的图片URL列表: {images}")
    return images
def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出,需要则流式输出
    2 根据state中的问题、重写问题、历史对话、提问商品（item_names）、 重排内容 组织prompt并调用llm 生成答案
    3 调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
    4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的 save_chat_message方法
    5 做最后一次push操作（主要是为了触发前端图片渲染)
    {
    "answer": "HAK 180 烫金机的操作面板位于. （大模型生成的纯文本）. ",
    "status": "completed",
    "image_urls": [
    "http: local-server/images/panel_view.jpg",
    "http: local-server/images/button_detail.jpg"
    ]
    }
    """
    function_name = sys._getframe().f_code.co_name
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    logger.info(f"开始执行节点: {function_name}")
    add_running_task(session_id, function_name, is_stream)
    # 1. 判断是否已有答案
    answer_exist = step1_check_existing_answer(state)
    if not answer_exist:
        # 2. 组织prompt
        prompt_template = step2_load_prompt(state)
        # 3. 输出答案
        answer = step3_generate_answer(state, prompt_template)
        # 4. 提取url
        image_urls = step4_extract_image_urls(state)
        # 5. 返回最终答案；即使没有图片，也用最终事件覆盖流式过程中的未格式化片段。
        if is_stream:
            push_to_session(session_id, SSEEvent.FINAL,
                             {"answer": answer, "status": "completed", "image_urls": image_urls})
        # 6. 保存到MongoDB
        save_chat_message(session_id=session_id, role="assistant", text=answer,
                          rewritten_query=state.get("rewritten_query", ""),
                          item_names=state.get("item_names", []),
                          image_urls=image_urls,
                          course_id=state.get("course_id", ""),
                          course_name=state.get("course_name", ""),
                          mode=state.get("mode", "qa"))
    add_done_task(session_id, function_name, is_stream)
    return state

if __name__ == "__main__":
    print("\n" + "="*50)
    print("> 启动 node_answer_output 本地测试")
    print("="*50)
    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
    {
    "chunk_id": "local_101",
    "source": "local",
    "title": "HAK 180 烫金机操作手册_v2.pdf",
    "score": 0.95,
    "text": """
    HAK 180 烫金机的操作面板位于机器正前方。
    开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
    具体的操作面板布局请参考下图：
    ![操作面板布局图](http: local-server/images/panel_view.jpg)
    如果是进行局部烫金，请调节侧面的旋钮。
    ![侧面旋钮细节](http: local-server/images/knob_detail.png)
    """
    },
    {
    "chunk_id": None,
    "source": "web",
    "title": "HAK 180 常见故障排除 - 官网",
    "score": 0.88,
    "url": "http: example.com/hak180_troubleshooting.jpeg", # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
    "text": "如果机器无法加热，请检查保险丝是否熔断. "
    },
    {
    "chunk_id": "local_102",
    "source": "local",
    "title": "安全注意事项",
    "score": 0.82,
    "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
    }
    ]
    # 模拟历史记录
    mock_history = [
    {"role": "user", "text": "你好，这款机器怎么用？"},
    {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
    {"role": "user", "text": "HAK 180 烫金机"}
    ]
    # 模拟输入状态
    mock_state = {
    "session_id": "test_answer_session_001",
    "original_query": "HAK 180 烫金机怎么操作？",
    "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
    "item_names": ["HAK180烫金机"],
    "history": mock_history,
    "reranked_docs": mock_reranked_docs,
    "is_stream": False, # 测试非流式
    # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
    "answer": None # 初始无答案
    }
    try:
    # 运行节点
        result = node_answer_output(mock_state)
        print("\n" + "="*50)
        print("> 测试结果摘要:")
        # # 1. 验证 Prompt 构建
        # if "prompt" in result:
        #     print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
        # # print(f"Prompt 预览:\n{result['prompt'][:200]}. ")
        # else:
        #     print("[FAIL] Prompt 未构建")
        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer}. ")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")
        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http: local-server/images/panel_view.jpg (来自 local_101)
        # 2. http: local-server/images/knob_detail.png (来自 local_101)
        # 3. http: example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url字段)
        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http: local-server/images/panel_view.jpg")
        print(" - http: local-server/images/knob_detail.png")
        print(" - http: example.com/hak180_troubleshooting.jpeg")
        print("="*50)
    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
