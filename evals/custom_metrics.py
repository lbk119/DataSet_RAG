from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "evals" / "reports" / "rag_outputs.jsonl"
DEFAULT_OUTPUT = ROOT / "evals" / "reports" / "custom_metrics.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def big_question_count(text: str) -> int:
    patterns = [
        r"(?m)^\s*[一二三四五六七八九十]+[、.．]",
        r"(?m)^\s*第[一二三四五六七八九十]+题",
    ]
    return max((len(re.findall(pattern, text or "")) for pattern in patterns), default=0)


def expected_hit(row: dict[str, Any], top_k: int = 5) -> bool:
    expected = row.get("expected_material_type")
    if not expected:
        return False
    docs = row.get("reranked_docs", []) or []
    return any(doc.get("material_type") == expected for doc in docs[:top_k])


def grouped_hit_rate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        groups.setdefault(key, []).append(row)
    return {
        key: {
            "cases": len(items),
            "expected_material_hit_at_5": sum(expected_hit(item, 5) for item in items) / len(items),
            "avg_context_count": mean([len(item.get("contexts", []) or []) for item in items]),
        }
        for key, items in sorted(groups.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute custom chain-level RAG metrics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        print("No rows found.")
        return 1

    total = len(rows)
    errors = [row for row in rows if row.get("error")]
    latencies = [row.get("latency_seconds", 0) for row in rows]
    context_counts = [len(row.get("contexts", [])) for row in rows]
    exam_rows = [row for row in rows if row.get("mode") == "exam" or row.get("expected_material_type") == "exam"]
    expected_hits = 0
    expected_total = 0
    course_leaks = 0
    doc_total = 0
    exam_ratios = []
    exam_structure_matches = []

    for row in rows:
        docs = row.get("reranked_docs", []) or []
        expected = row.get("expected_material_type")
        if expected:
            expected_total += 1
            if expected_hit(row, 5):
                expected_hits += 1
        for doc in docs:
            doc_total += 1
            if doc.get("course_name") and row.get("course_name") and doc.get("course_name") != row.get("course_name"):
                course_leaks += 1
        if row in exam_rows:
            if docs:
                exam_ratios.append(sum(1 for doc in docs if doc.get("material_type") == "exam") / len(docs))
            answer_count = big_question_count(row.get("answer", ""))
            reference_count = big_question_count(row.get("reference", ""))
            if reference_count:
                exam_structure_matches.append(1.0 if answer_count == reference_count else 0.0)

    report = {
        "total_cases": total,
        "error_rate": len(errors) / total,
        "avg_latency_seconds": mean(latencies) if latencies else 0,
        "avg_context_count": mean(context_counts) if context_counts else 0,
        "expected_material_hit_at_5": expected_hits / expected_total if expected_total else None,
        "course_leak_rate": course_leaks / doc_total if doc_total else 0,
        "exam_context_ratio": mean(exam_ratios) if exam_ratios else None,
        "exam_big_question_count_match": mean(exam_structure_matches) if exam_structure_matches else None,
        "by_eval_type": grouped_hit_rate(rows, "eval_type"),
        "by_expected_topic": grouped_hit_rate(rows, "expected_topic"),
        "by_expected_source": grouped_hit_rate(rows, "expected_source"),
        "failed_case_ids": [row.get("id") for row in errors],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved custom metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
