import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple

if __package__ in (None, ""):
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task, add_done_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger # 项目统一日志工具，核心替换print

# - 配置参数 (Configuration) -
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500
SECTION_MAX_CONTENT_LENGTH = 1000
SECTION_MIN_CONTENT_LENGTH = 350

EXAM_YEAR_PATTERN = re.compile(r"(19\d{2}|20\d{2})")
EXAM_SCORE_PATTERN = re.compile(r"(\d{1,3})\s*(?:分|points?|pts?)", re.IGNORECASE)
EXAM_QUESTION_PATTERNS = [
    re.compile(r"(?m)^\s*([一二三四五六七八九十]+)[、.．:：\s]+([^\n]{0,80})"),
    re.compile(r"(?m)^\s*第\s*([一二三四五六七八九十\d]+)\s*[题題][、.．:：\s]*([^\n]{0,80})"),
    re.compile(r"(?m)^\s*(\d{1,2})[、.．:：\s]+([^\n]{0,80})"),
]
EXAM_TYPE_KEYWORDS = [
    "选择", "填空", "判断", "简答", "计算", "证明", "编程", "分析", "设计", "应用", "综合", "作图", "问答", "解答",
]
EXAM_TOPIC_KEYWORDS = [
    "插值", "Hermite", "埃尔米特", "拉格朗日", "牛顿", "差商", "拟合", "最小二乘",
    "数值积分", "数值微分", "梯形", "辛普森", "Simpson", "方程求根", "二分法",
    "牛顿法", "迭代", "收敛", "误差", "线性方程组", "高斯", "矩阵", "特征值",
    "微分方程", "Euler", "龙格", "Runge", "稳定性",
]
QUESTION_SECTION_PATTERN = re.compile(
    r"(?m)^\s*((?:第\s*)?[一二三四五六七八九十\d]{1,3}\s*(?:题|題)|[一二三四五六七八九十\d]{1,3}\s*[、.．])"
)


def _compact_exam_text(text: str, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _first_match(pattern: re.Pattern, text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1) if match else ""


def _infer_exam_question(text: str) -> tuple[str, str]:
    for pattern in EXAM_QUESTION_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return match.group(1).strip(), _compact_exam_text(match.group(2), 80)
    return "", ""


def _infer_exam_question_type(text: str) -> str:
    lowered = (text or "").lower()
    for keyword in EXAM_TYPE_KEYWORDS:
        if keyword.lower() in lowered:
            return keyword
    return ""


def _infer_exam_topics(text: str, limit: int = 5) -> str:
    lowered = (text or "").lower()
    topics = []
    for keyword in EXAM_TOPIC_KEYWORDS:
        if keyword.lower() in lowered and keyword not in topics:
            topics.append(keyword)
    return "、".join(topics[:limit])


def infer_topics(text: str, limit: int = 5) -> str:
    return _infer_exam_topics(text, limit=limit)


def split_question_sections(sections: List[Dict[str, Any]], material_type: str) -> List[Dict[str, Any]]:
    if material_type not in {"exam", "exam_answer", "homework"}:
        return sections

    result = []
    for section in sections:
        content = section.get("content", "") or ""
        matches = list(QUESTION_SECTION_PATTERN.finditer(content))
        if len(matches) < 2:
            result.append(section)
            continue

        prefix = content[: matches[0].start()].strip()
        for index, match in enumerate(matches, start=1):
            start = match.start()
            end = matches[index].start() if index < len(matches) else len(content)
            block = content[start:end].strip()
            if not block:
                continue
            title_text = block.splitlines()[0].strip()[:80]
            if prefix and index == 1:
                block = prefix + "\n\n" + block
            item = dict(section)
            item["title"] = f"{section.get('title') or section.get('file_title', '')} / {title_text}".strip(" /")
            item["parent_title"] = section.get("title") or section.get("parent_title") or section.get("file_title", "")
            item["content"] = block
            item["part"] = index
            result.append(item)
    return result


def enrich_common_chunk_metadata(chunk: Dict[str, Any], state: ImportGraphState) -> None:
    scan_text = "\n".join([
        str(chunk.get("file_title") or state.get("file_title", "")),
        str(chunk.get("title") or ""),
        str(chunk.get("content") or "")[:2000],
    ])
    topics = infer_topics(scan_text)
    chunk["topics"] = topics
    chunk["primary_topic"] = topics.split("、")[0] if topics else ""


def enrich_exam_chunk_metadata(chunk: Dict[str, Any], state: ImportGraphState) -> None:
    if chunk.get("material_type") != "exam":
        return

    file_title = str(chunk.get("file_title") or state.get("file_title", ""))
    title = str(chunk.get("title") or "")
    content = str(chunk.get("content") or "")
    scan_text = f"{file_title}\n{title}\n{content[:2000]}"
    question_no, question_title = _infer_exam_question(scan_text)
    score = _first_match(EXAM_SCORE_PATTERN, scan_text)
    is_reference_answer = bool(re.search(r"(参考答案|答案|解析|评分|answer|solution)", scan_text, re.IGNORECASE))

    chunk["exam_year"] = _first_match(EXAM_YEAR_PATTERN, scan_text)
    chunk["exam_question_no"] = question_no
    chunk["exam_question_title"] = question_title
    chunk["exam_question_type"] = _infer_exam_question_type(scan_text)
    chunk["exam_score"] = int(score) if score.isdigit() else 0
    chunk["exam_topics"] = _infer_exam_topics(scan_text)
    chunk["is_reference_answer"] = is_reference_answer

def step1_load_input(state: ImportGraphState) -> Tuple[str, str]:
    """
    步骤1：加载输入数据
    从状态字典中获取Markdown内容和文件标题，进行基本的有效性检查
    Args:
        state: ImportGraphState对象，必须包含md_content和file_title字段
    Returns:
        content: Markdown文本内容（字符串）
        file_title: 文件标题（字符串）
    """
    content = state.get("md_content", "")
    content = content.replace("\r\n", "\n").replace("\r", "\n")  # 统一换行符，避免不同系统导致的切分问题
    file_title = state.get("file_title", "Unknown File")
    if not content:
        logger.warning("步骤1 - 加载输入数据：Markdown内容为空，后续处理可能无效")
    else:
        logger.info(f"步骤1 - 加载输入数据：成功加载Markdown内容，长度{len(content)}字符，文件标题：{file_title}")
    return content, file_title

def step2_initial_split(content: str,file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    步骤2：按Markdown标题初步切分
    Args:
        content: Markdown文本内容（字符串）
        file_title: 文件标题（字符串）
    Returns:
        sections: 初切后的章节列表，每个章节为字典，包含title/content/parent_title
        title_count: 识别到的有效标题数量
        lines_count: Markdown原始文本总行数
    """
    # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
    # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
    # # 1,6}：匹配1-6个#（对应MD1-6级标题）
    # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
    # .+：标题文字至少1个字符（避免空标题）
    title_pattern = re.compile(r"^\s*(#{1,6})\s+.+", re.MULTILINE)
    lines = content.split("\n")
    sections = []
    current_title = ""
    current_lines = [] # 当前章节的行缓存
    title_count = 0
    lines_count = len(lines)
    is_in_code_block = False  # 代码块标记，避免切分器误将代码中的#识别为标题
    for line in lines:
        line = line.strip()  
        # 代码块检测：遇到```或~~~切换状态，代码块内不识别标题
        if line.startswith("```") or line.startswith("~~~"):
            is_in_code_block = not is_in_code_block
            current_lines.append(line)  # 代码块标记行也加入当前章节内容
            continue
        title_match = (not is_in_code_block) and title_pattern.match(line)
        if title_match:
            # 遇到新标题，先保存上一个章节（如果有内容）
            if current_lines:
                sections.append({"title": current_title, "content": "\n".join(current_lines), "file_title": file_title})
            current_lines = [line]  # 新章节内容从当前行开始
            current_title = line
            title_count += 1
        else:
            current_lines.append(line)

    # 保存最后一个章节（如果有内容）
    if current_lines:
        sections.append({"title": current_title, "content": "\n".join(current_lines), "file_title": file_title})

    logger.info(f"步骤2 - 按Markdown标题初步切分：识别到{title_count}个有效标题，原始文本共{lines_count}行，初切分成{len(sections)}个章节")
    return sections, title_count, lines_count

def split_long_section(section: Dict[str, Any], max_length: int) -> List[Dict[str, Any]]:
    """
    对单个章节进行长度检查，超过max_length则使用RecursiveCharacterTextSplitter进行二次切分
    功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
    切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
    Args:
        section: 章节字典，包含title/content/file_title
        max_length: 单个Chunk最大字符长度，超过则触发切分
    Returns:
        split_sections: 切分后的章节列表，每个章节为字典，包含title/content/file_title
    """
    content = section["content"] or ""
    if len(content) <= max_length:
        return [section]  # 不需要切分，直接返回原章节

    content = content.replace("\r\n", "\n").replace("\r", "\n")  # 统一换行符，避免切分问题
    # 提取章节标题，用于组装子Chunk前缀（保留标题上下文）
    title = section.get("title", "") or ""
    # 标题前缀：带空行分隔，与正文区分开
    prefix = f"{title}\n\n" if title else ""
    # 计算正文可用长度：总长度 - 标题前缀长度（避免标题占满Chunk额度）
    available_len = max_length - len(prefix)
    # 极端情况：标题长度超过阈值，无法切分，返回原章节
    if available_len < 0:
        logger.warning(f"章节标题过长，无法切分：{title[:20]}. ")
        return [section]
    # 使用LangChain的RecursiveCharacterTextSplitter进行切分，保持原有标题和文件名
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_length, chunk_overlap=100,separators=["\n\n", "\n", " ", ""])  # 从粗到细的切分规则
    split_sections = []
    for i, chunk in enumerate(text_splitter.split_text(content),start=1):
        text = chunk.strip()  # 去除切分后文本的首尾空白，避免无效字符占用长度
        title = f"{section.get("title", "")}_{i}"  # 子Chunk标题：原标题+序号，保持层级关系
        parent_title = section.get("title", "")  # 父标题保持不变，便于后续合并/检索
        file_title = section.get("file_title", "")  # 文件标题保持不变
        split_sections.append({
            "title": title,
            "parent_title": parent_title,
            "content": text,
            "file_title": file_title,
            "part": i
        })
    logger.info(f"步骤4 - 长切短合处理：章节'{section['title']}'长度{len(content)}字符，切分成{len(split_sections)}个部分")
    return split_sections

def merge_short_sections(sections: List[Dict[str, Any]], min_length: int) -> List[Dict[str, Any]]:
    """
    对同父标题的短章节进行合并，减少碎片化
    功能：同一父标题下的章节如果内容过短（小于min_length），则合并到前一个章节中，增强语义完整性
    Args:
        sections: 章节列表，每个章节为字典，包含title/content/parent_title/file_title
        min_length: 短Chunk合并阈值，同父标题的短Chunk会被合并
    Returns:
        merged_sections: 合并后的章节列表，每个章节为字典，包含title/content/parent_title/file_title
    """
    if not sections:
        return []
    merged_sections = []
    current_section = sections[0]  # 初始化当前章节为第一个
    for next_section in sections[1:]:
        # 判断是否需要合并：父标题存在且同父标题且内容长度不足
        is_same_parent = current_section.get("parent_title") and (current_section.get("parent_title") == next_section.get("parent_title"))
        is_short = len(current_section.get("content", "")) < min_length
        if is_same_parent and is_short:
            # 合并内容：用换行分隔，保持原有标题和文件名不变
            current_section["content"] = f"{current_section.get('content', '')}\n\n{next_section.get('content', '')}"
            current_section["part"] = next_section.get("part", 1)  # 更新part为合并后的最新值
        else:
            # 不满足合并条件，先保存当前章节，再切换到下一个章节
            merged_sections.append(current_section)
            current_section = next_section
    merged_sections.append(current_section)  # 添加最后一个章节
    return merged_sections
def step4_split_and_merge(sections: List[Dict[str, Any]], max_length: int, min_length: int) -> List[Dict[str, Any]]:
    """
    步骤4：长切短合处理
    对初切后的章节列表进行长度检查，超过max_length的章节会被进一步切分；同时对同父标题的短章节进行合并，减少碎片化
    Args:
        sections: 初切后的章节列表，每个章节为字典，包含title/content/file_title
        max_length: 单个Chunk最大字符长度，超过则触发二次切分
        min_length: 短Chunk合并阈值，同父标题的短Chunk会被合并
    Returns:
        processed_chunks: 处理后的Chunk列表，每个Chunk为字典，包含title/content/file_title
    """
    # 阶段1：切分超长章节 → 所有章节长度控制在max_len内
    split_sections = []
    for section in sections:
        # extend 的作用就是： 把另一个列表（或可迭代对象）里的“元素”，一个个拆出来，直接追加到当前列表的尾部
        split_sections.extend(split_long_section(section, max_length))
    logger.info(f"步骤4 - 长切短合处理：阶段1完成，切分超长章节后共{len(split_sections)}个章节")
    # 阶段2：合并同父标题的短章节 → 减少碎片化，增强语义完整性
    result_sections = merge_short_sections(split_sections, min_length)
    logger.info(f"步骤4 - 长切短合处理：阶段2完成，合并短章节后共{len(result_sections)}个章节")
    # 阶段3：父标题兜底 → 适配Milvus向量库schema（parent_title为必填字段）
    # 兜底规则：无parent_title则用自身title，title也无则填空字符串
    for section in result_sections:
        if not section.get("parent_title"):
            section["parent_title"] = section.get("title", "") or ""
        if section.get("part") is None:
            section["part"] = 1
    return result_sections

def step5_backup_results(state: ImportGraphState, chunks: List[Dict[str, Any]]):
    """
    步骤5：结果备份
    可选地将结果写入本地文件，便于调试和验证
    Args:
        state: ImportGraphState对象，包含task_id/local_dir/file_title等字段
        chunks: 处理后的Chunk列表，每个Chunk为字典，包含title/content/parent_title/file_title
    """
    # 可选：将结果写入本地文件，便于调试和验证（根据实际需求决定是否启用）
    local_dir = state.get("local_dir", "")
    backup_file_path = os.path.join(local_dir, "chunks.json") 
    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"步骤5 - 结果备份：处理后的Chunk列表已写入{backup_file_path}")

def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    整体流程：加载输入→按MD标题初切→无标题兜底→长切短合→结果备份
    核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
    后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等
    :param state: 项目状态字典（ImportGraphState），必须包含md_content/task_id；可选local_dir/max_content_length/file_title
    :return: 更新后的状态字典，新增chunks键（存储最终处理后的Chunk列表，每个Chunk为含title/content/parent_title的字典）
    """
    func_name = sys._getframe().f_code.co_name  # 获取当前函数名，便于日志记录
    logger.info(f"开始执行{func_name}")
    add_running_task(state["task_id"], func_name)  # 记录当前任务状态，便于监控和调度

    try:
        # 1. 加载输入数据：从状态中获取MD内容、文件标题
        content,file_title = step1_load_input(state)
        # 2. 按MD标题初步切分：以Markdown标题层级（#/##/###）为基础，自动跳过代码块内的伪标题，切分成初始Chunk列表
        # 输出：初切后的章节列表、识别到的有效标题数量、MD原始文本总行数（为后续统计/日志使用）
        sections, title_count, lines_count = step2_initial_split(content, file_title)
        # 3. 无标题兜底处理：如果初切后没有有效标题，则将整个文档作为一个Chunk，标题使用文件名（或默认值）
        if title_count == 0:
            logger.info(f"{func_name} - 未识别到有效标题，启用无标题兜底处理")
            sections = [{"title": "无标题", "content": content, "file_title": file_title}]
        material_type = state.get("material_type", "other")
        sections = split_question_sections(sections, material_type)
        # 4. 长切短合处理：对初切后的Chunk列表进行长度检查，超过max_content_length的Chunk会被进一步切分；同时对同父标题的短Chunk进行合并，减少碎片化
        max_len = DEFAULT_MAX_CONTENT_LENGTH
        min_len = MIN_CONTENT_LENGTH
        if material_type in {"textbook", "slides", "courseware", "other"}:
            max_len = SECTION_MAX_CONTENT_LENGTH
            min_len = SECTION_MIN_CONTENT_LENGTH
        processed_chunks = step4_split_and_merge(sections, max_len, min_len)
        for chunk in processed_chunks:
            chunk["course_id"] = state.get("course_id", "")
            chunk["course_name"] = state.get("course_name", "")
            chunk["material_type"] = material_type
            enrich_common_chunk_metadata(chunk, state)
            enrich_exam_chunk_metadata(chunk, state)
        # 5. 结果备份：将处理后的Chunk列表存储到状态字典中，便于后续节点使用
        state["chunks"] = processed_chunks
        step5_backup_results(state, processed_chunks)
        add_done_task(state["task_id"], func_name)  # 记录当前任务完成状态，便于监控和调度
        logger.info(f"{func_name} - 文档切分完成，共生成{len(processed_chunks)}个Chunk")
    except Exception as e:
        logger.error(f"{func_name} - 文档切分失败，错误信息：{e}")
        raise e
    return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")
    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)
    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
    # 构造测试状态对象，模拟流程入参
        test_state = {
        "md_path": test_md_path,
        "task_id": "test_task_123456",
        "md_content": "",
        "file_title": "hak180产品安全手册",
        "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n= 开始执行文档切分节点集成测试 = ")
        logger.info("> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk")
        logger.info(f"最终生成的Chunk列表：{final_chunks}")
