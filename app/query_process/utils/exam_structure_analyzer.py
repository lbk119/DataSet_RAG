from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


QUESTION_PATTERNS = [
    re.compile(r"(?m)^\s*([一二三四五六七八九十]+)[、.．]\s*([^\n]{0,80})"),
    re.compile(r"(?m)^\s*(第[一二三四五六七八九十]+题)\s*[:：、.．]?\s*([^\n]{0,80})"),
    re.compile(r"(?m)^\s*(\d+)[、.．]\s*([^\n]{0,80})"),
]
SCORE_PATTERN = re.compile(r"(\d{1,3})\s*(?:分|points?|pts?)", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(20\d{2}|19\d{2})")
TYPE_KEYWORDS = [
    "选择",
    "填空",
    "判断",
    "简答",
    "计算",
    "证明",
    "编程",
    "分析",
    "设计",
    "应用",
    "综合",
    "作图",
    "问答",
    "解答",
]
TOPIC_KEYWORDS = [
    "插值",
    "拟合",
    "数值积分",
    "数值微分",
    "方程求根",
    "牛顿",
    "拉格朗日",
    "Hermite",
    "埃尔米特",
    "Euler",
    "Runge",
    "龙格",
    "微分方程",
    "线性方程组",
    "迭代",
    "误差",
    "稳定性",
    "收敛",
    "矩阵",
    "特征值",
    "极限",
    "导数",
    "积分",
    "级数",
    "概率",
    "图",
    "树",
    "排序",
    "查找",
    "动态规划",
    "数据库",
    "操作系统",
    "进程",
    "线程",
]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _infer_question_type(text: str) -> str:
    for keyword in TYPE_KEYWORDS:
        if keyword in text:
            return keyword
    return "未明确"


def _extract_topics(text: str, limit: int = 5) -> list[str]:
    found = []
    for keyword in TOPIC_KEYWORDS:
        if keyword.lower() in text.lower() and keyword not in found:
            found.append(keyword)
    return found[:limit]


def _extract_questions(text: str) -> list[dict[str, Any]]:
    questions = []
    seen = set()
    for pattern in QUESTION_PATTERNS:
        for match in pattern.finditer(text):
            order = match.group(1)
            title = _clean(match.group(2))
            key = (order, title)
            if key in seen:
                continue
            seen.add(key)
            start = match.start()
            window = text[start : start + 500]
            score_match = SCORE_PATTERN.search(window)
            questions.append(
                {
                    "order": order,
                    "title": title,
                    "type": _infer_question_type(title + " " + window[:180]),
                    "score": int(score_match.group(1)) if score_match else None,
                    "topics": _extract_topics(window),
                }
            )
    return questions


def _parse_position(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    chinese_digits = "一二三四五六七八九十"
    if text in chinese_digits:
        return chinese_digits.index(text) + 1
    return None


def _doc_topics(doc: dict[str, Any], fallback_text: str) -> list[str]:
    topics = doc.get("exam_topics") or ""
    if isinstance(topics, list):
        return [str(topic) for topic in topics if str(topic).strip()]
    if isinstance(topics, str) and topics.strip():
        return [topic.strip() for topic in re.split(r"[、,，;\s]+", topics) if topic.strip()]
    return _extract_topics(fallback_text)


def analyze_exam_structure(docs: list[dict[str, Any]]) -> str:
    exam_docs = [doc for doc in docs if doc.get("material_type") == "exam"]
    if not exam_docs:
        return "未检索到往年试卷切片，试卷结构分析依据不足。"

    exam_count = len(exam_docs)
    years = Counter()
    question_count_by_doc = []
    type_counter = Counter()
    score_counter = Counter()
    topic_counter = Counter()
    by_position: dict[int, dict[str, Counter]] = defaultdict(lambda: {"types": Counter(), "topics": Counter(), "scores": Counter()})

    for doc in exam_docs:
        text = doc.get("text", "") or ""
        title = doc.get("title", "") or ""
        meta_year = str(doc.get("exam_year") or "").strip()
        if meta_year:
            years[meta_year] += 1
        for year in YEAR_PATTERN.findall(title + " " + text[:300]):
            years[year] += 1

        meta_position = _parse_position(doc.get("exam_question_no"))
        meta_type = str(doc.get("exam_question_type") or "").strip()
        meta_score = doc.get("exam_score")
        try:
            meta_score = int(meta_score) if meta_score not in (None, "") else None
        except (TypeError, ValueError):
            meta_score = None
        meta_topics = _doc_topics(doc, text)
        if meta_position:
            if meta_type:
                type_counter[meta_type] += 1
                by_position[meta_position]["types"][meta_type] += 1
            if meta_score:
                score_counter[meta_score] += 1
                by_position[meta_position]["scores"][meta_score] += 1
            for topic in meta_topics:
                topic_counter[topic] += 1
                by_position[meta_position]["topics"][topic] += 1

        questions = _extract_questions(text)
        if questions:
            question_count_by_doc.append(len(questions))
        for index, question in enumerate(questions, start=1):
            q_type = question["type"]
            type_counter[q_type] += 1
            by_position[index]["types"][q_type] += 1
            if question["score"] is not None:
                score_counter[question["score"]] += 1
                by_position[index]["scores"][question["score"]] += 1
            for topic in question["topics"]:
                topic_counter[topic] += 1
                by_position[index]["topics"][topic] += 1

    common_count = Counter(question_count_by_doc).most_common(1)
    predicted_count = common_count[0][0] if common_count else None
    lines = [
        "【试卷结构分析中间结果】",
        f"- 已纳入分析的往年试卷切片数：{exam_count}",
        f"- 识别到的年份：{', '.join(year for year, _ in years.most_common()) if years else '未明确'}",
        f"- 推测常见大题数量：{predicted_count if predicted_count else '未能稳定识别'}",
        f"- 高频题型：{', '.join(f'{name}({count})' for name, count in type_counter.most_common(8)) if type_counter else '未明确'}",
        f"- 常见分值：{', '.join(f'{score}分({count})' for score, count in score_counter.most_common(8)) if score_counter else '未明确'}",
        f"- 高频考点：{', '.join(f'{topic}({count})' for topic, count in topic_counter.most_common(12)) if topic_counter else '未明确'}",
    ]

    if by_position:
        lines.append("")
        lines.append("【按大题位置统计】")
        max_position = predicted_count or max(by_position)
        for index in range(1, max_position + 1):
            data = by_position.get(index)
            if not data:
                continue
            q_type = data["types"].most_common(1)[0][0] if data["types"] else "未明确"
            topics = ", ".join(topic for topic, _ in data["topics"].most_common(5)) or "未明确"
            score = f"{data['scores'].most_common(1)[0][0]}分" if data["scores"] else "未明确"
            lines.append(f"- 第{index}大题：常见题型={q_type}；常见分值={score}；常考方向={topics}")

    lines.append("")
    lines.append("请将以上结构分析作为出卷最高优先级依据；若与原始试卷切片冲突，以原始试卷切片为准。")
    return "\n".join(lines)
