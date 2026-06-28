import base64
import io
import mimetypes
import uuid
from pathlib import Path
from typing import Iterable, List, Dict

from langchain_core.messages import HumanMessage

from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".java", ".cpp", ".c", ".h", ".html", ".css", ".js"}
MAX_ATTACHMENT_TEXT_CHARS = 12000
MAX_IMAGE_SIDE = 1600
VL_TIMEOUT_SECONDS = 90


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "gbk", "gb2312"):
        try:
            return path.read_text(encoding=encoding)[:MAX_ATTACHMENT_TEXT_CHARS]
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")[:MAX_ATTACHMENT_TEXT_CHARS]


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
            raw = output.getvalue()
            mime_type = "image/jpeg"
    except Exception as e:
        logger.warning(f"image resize failed, sending original file: {path}, error={e}")
        raw = path.read_bytes()
    data = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def summarize_image(path: Path, question: str = "") -> str:
    llm_client = get_llm_client(lm_config.lv_model, timeout=VL_TIMEOUT_SECONDS)
    prompt = (
        "你是大学专业课助教。请仔细阅读用户上传的图片，提取与解题或学习有关的信息。"
        "如果图片是题目、公式、表格、图像或代码，请尽量转写关键内容；"
        "如果用户问题给出了具体要求，请围绕该要求分析。\n\n"
        f"用户问题：{question or '未提供文字问题'}"
    )
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_data_url(path)}},
            ]
        )
    ]
    try:
        response = llm_client.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"query image attachment analysis failed: {path}, model={lm_config.lv_model}, error={e}")
        return (
            f"Image analysis failed. Vision model={lm_config.lv_model}. "
            f"File={path.name}. Error={e}. "
            "Please ask the user to retry with a smaller/clearer image or type the problem text."
        )


def parse_pdf_with_mineru(path: Path, work_dir: Path) -> str:
    """
    Query-side lightweight PDF parsing. Reuses the import-side MinerU parser when configured.
    """
    from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md

    work_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "task_id": f"query-attachment-{uuid.uuid4().hex[:8]}",
        "pdf_path": str(path),
        "local_dir": str(work_dir),
        "md_path": "",
        "md_content": "",
    }
    node_pdf_to_md(state)
    md_path = state.get("md_path", "")
    if md_path and Path(md_path).exists():
        return _read_text_file(Path(md_path))
    return state.get("md_content", "")[:MAX_ATTACHMENT_TEXT_CHARS]


def analyze_attachment(path: Path, question: str = "", work_dir: Path | None = None) -> Dict[str, str]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        text = summarize_image(path, question)
        return {"name": path.name, "type": "image", "text": text}
    if suffix in TEXT_EXTENSIONS:
        return {"name": path.name, "type": "text", "text": _read_text_file(path)}
    if suffix == ".pdf":
        try:
            text = parse_pdf_with_mineru(path, work_dir or path.parent)
        except Exception as e:
            logger.warning(f"查询附件PDF解析失败: {path}, error={e}")
            text = (
                "PDF即时解析失败。建议将该 PDF 通过导入侧加入课程知识库后再提问；"
                f"失败原因：{e}"
            )
        return {"name": path.name, "type": "pdf", "text": text[:MAX_ATTACHMENT_TEXT_CHARS]}
    return {"name": path.name, "type": "unsupported", "text": "暂不支持该附件类型的即时解析。"}


def build_attachment_context(paths: Iterable[Path], question: str = "", work_dir: Path | None = None) -> str:
    parts: List[str] = []
    for index, path in enumerate(paths, start=1):
        try:
            item = analyze_attachment(path, question=question, work_dir=work_dir)
        except Exception as e:
            logger.warning(f"query attachment analysis failed: {path}, error={e}")
            item = {"name": path.name, "type": "error", "text": f"Attachment analysis failed: {e}"}
        parts.append(
            f"附件 {index}: {item['name']} ({item['type']})\n"
            f"{item['text']}"
        )
    return "\n\n".join(parts).strip()
