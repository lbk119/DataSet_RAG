from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAG_OUTPUTS = ROOT / "evals" / "reports" / "rag_outputs.jsonl"
DEFAULT_CUSTOM_METRICS = ROOT / "evals" / "reports" / "custom_metrics.json"
DEFAULT_RAGAS_REPORT = ROOT / "evals" / "reports" / "ragas_report.csv"
DEFAULT_OUTPUT_JSON = ROOT / "evals" / "reports" / "eval_summary.json"
DEFAULT_OUTPUT_MD = ROOT / "evals" / "reports" / "eval_summary.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def round_value(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": round(mean(values), 4),
        "p50": round(median(values), 4),
        "p90": round(percentile(values, 0.9), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def material_hit(row: dict[str, Any], top_k: int) -> bool:
    expected = row.get("expected_material_type")
    if not expected:
        return False
    docs = row.get("reranked_docs", []) or []
    return any(doc.get("material_type") == expected for doc in docs[:top_k])


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    errors = [row for row in rows if row.get("error")]
    answered = [row for row in rows if row.get("answer") and not row.get("error")]
    latencies = [safe_float(row.get("latency_seconds")) or 0.0 for row in rows]
    context_counts = [len(row.get("contexts", []) or []) for row in rows]

    doc_total = 0
    course_leaks = 0
    top_material_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    score_values: list[float] = []
    for row in rows:
        docs = row.get("reranked_docs", []) or []
        for doc in docs[:5]:
            material_type = doc.get("material_type") or "unknown"
            top_material_counter[material_type] += 1
            source_counter[doc.get("source") or "unknown"] += 1
            score = safe_float(doc.get("score"))
            if score is not None:
                score_values.append(score)
        for doc in docs:
            doc_total += 1
            if doc.get("course_name") and row.get("course_name") and doc.get("course_name") != row.get("course_name"):
                course_leaks += 1

    expected_rows = [row for row in rows if row.get("expected_material_type")]
    hit_at = {
        "hit_at_1": sum(material_hit(row, 1) for row in expected_rows) / len(expected_rows) if expected_rows else None,
        "hit_at_3": sum(material_hit(row, 3) for row in expected_rows) / len(expected_rows) if expected_rows else None,
        "hit_at_5": sum(material_hit(row, 5) for row in expected_rows) / len(expected_rows) if expected_rows else None,
    }

    return {
        "total_cases": total,
        "answered_cases": len(answered),
        "error_cases": len(errors),
        "error_rate": len(errors) / total if total else 0.0,
        "zero_context_rate": sum(1 for count in context_counts if count == 0) / total if total else 0.0,
        "latency_seconds": summarize_numeric(latencies),
        "context_count": summarize_numeric([float(count) for count in context_counts]),
        "retrieval_score": summarize_numeric(score_values),
        "expected_material": {key: round_value(value) if value is not None else None for key, value in hit_at.items()},
        "course_leak_rate": course_leaks / doc_total if doc_total else 0.0,
        "top5_material_distribution": dict(top_material_counter.most_common()),
        "top5_source_distribution": dict(source_counter.most_common()),
        "failed_case_ids": [row.get("id") for row in errors],
    }


def group_rows(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "unknown")].append(row)
    return {key: summarize_rows(value) for key, value in sorted(grouped.items())}


def summarize_ragas(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    ignored = {"user_input", "response", "retrieved_contexts", "reference", "id"}
    metric_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key in ignored:
                continue
            number = safe_float(value)
            if number is not None:
                metric_values[key].append(number)

    summary = {}
    for metric, values in sorted(metric_values.items()):
        summary[metric] = summarize_numeric(values)
    return {
        "available": True,
        "total_rows": len(rows),
        "metrics": summary,
    }


def format_percent(value: Any) -> str:
    if value is None:
        return "-"
    number = safe_float(value)
    if number is None:
        return str(value)
    return f"{number * 100:.2f}%"


def format_number(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return str(value)
    return f"{number:.4f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def build_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    custom = summary.get("custom_metrics") or {}
    ragas = summary.get("ragas") or {}

    lines = [
        "# RAG Evaluation Summary",
        "",
        "## Overall",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Total cases", overall["total_cases"]],
                ["Answered cases", overall["answered_cases"]],
                ["Error rate", format_percent(overall["error_rate"])],
                ["Zero context rate", format_percent(overall["zero_context_rate"])],
                ["Avg latency seconds", format_number(overall["latency_seconds"]["avg"])],
                ["P90 latency seconds", format_number(overall["latency_seconds"]["p90"])],
                ["Avg context count", format_number(overall["context_count"]["avg"])],
                ["Expected material hit@1", format_percent(overall["expected_material"]["hit_at_1"])],
                ["Expected material hit@3", format_percent(overall["expected_material"]["hit_at_3"])],
                ["Expected material hit@5", format_percent(overall["expected_material"]["hit_at_5"])],
                ["Course leak rate", format_percent(overall["course_leak_rate"])],
            ],
        ),
        "",
    ]

    if custom:
        lines.extend(
            [
                "## Custom Metrics",
                "",
                markdown_table(
                    ["Metric", "Value"],
                    [[key, format_number(value) if isinstance(value, (float, int)) else value] for key, value in custom.items()],
                ),
                "",
            ]
        )

    lines.extend(["## Group By Mode", ""])
    mode_rows = []
    for key, data in summary["by_mode"].items():
        mode_rows.append(
            [
                key,
                data["total_cases"],
                format_percent(data["error_rate"]),
                format_number(data["latency_seconds"]["avg"]),
                format_number(data["context_count"]["avg"]),
                format_percent(data["expected_material"]["hit_at_5"]),
            ]
        )
    lines.extend([markdown_table(["Mode", "Cases", "Error Rate", "Avg Latency", "Avg Contexts", "Hit@5"], mode_rows), ""])

    lines.extend(["## Group By Expected Material", ""])
    material_rows = []
    for key, data in summary["by_expected_material_type"].items():
        material_rows.append(
            [
                key,
                data["total_cases"],
                format_percent(data["error_rate"]),
                format_number(data["latency_seconds"]["avg"]),
                format_number(data["context_count"]["avg"]),
                format_percent(data["expected_material"]["hit_at_5"]),
            ]
        )
    lines.extend([markdown_table(["Expected Material", "Cases", "Error Rate", "Avg Latency", "Avg Contexts", "Hit@5"], material_rows), ""])

    def append_group_table(title: str, group_key: str, first_col: str) -> None:
        grouped = summary.get(group_key) or {}
        if not grouped:
            return
        rows = []
        for key, data in grouped.items():
            rows.append(
                [
                    key,
                    data["total_cases"],
                    format_percent(data["error_rate"]),
                    format_number(data["latency_seconds"]["avg"]),
                    format_number(data["context_count"]["avg"]),
                    format_percent(data["expected_material"]["hit_at_5"]),
                ]
            )
        lines.extend([f"## {title}", "", markdown_table([first_col, "Cases", "Error Rate", "Avg Latency", "Avg Contexts", "Hit@5"], rows), ""])

    append_group_table("Group By Eval Type", "by_eval_type", "Eval Type")
    append_group_table("Group By Expected Topic", "by_expected_topic", "Expected Topic")
    append_group_table("Group By Expected Source", "by_expected_source", "Expected Source")

    lines.extend(["## Retrieval Distribution", ""])
    lines.extend(
        [
            markdown_table(
                ["Material Type", "Top5 Count"],
                [[key, value] for key, value in overall["top5_material_distribution"].items()] or [["-", "-"]],
            ),
            "",
        ]
    )

    lines.extend(["## Ragas", ""])
    if not ragas.get("available"):
        lines.extend(["Ragas report not found. Run `uv run python evals/ragas_eval.py` first if you want LLM-judged metrics.", ""])
    else:
        ragas_rows = []
        for metric, data in ragas["metrics"].items():
            ragas_rows.append(
                [
                    metric,
                    format_number(data["avg"]),
                    format_number(data["p50"]),
                    format_number(data["p90"]),
                    format_number(data["min"]),
                    format_number(data["max"]),
                ]
            )
        lines.extend([markdown_table(["Metric", "Avg", "P50", "P90", "Min", "Max"], ragas_rows), ""])

    failed_case_ids = overall.get("failed_case_ids") or []
    if failed_case_ids:
        lines.extend(["## Failed Cases", "", ", ".join(str(item) for item in failed_case_ids), ""])

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize RAG, custom, and Ragas evaluation outputs.")
    parser.add_argument("--rag-outputs", type=Path, default=DEFAULT_RAG_OUTPUTS)
    parser.add_argument("--custom-metrics", type=Path, default=DEFAULT_CUSTOM_METRICS)
    parser.add_argument("--ragas-report", type=Path, default=DEFAULT_RAGAS_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = read_jsonl(args.rag_outputs)
    if not rows:
        print(f"No RAG output rows found: {args.rag_outputs}")
        return 1

    summary = {
        "inputs": {
            "rag_outputs": str(args.rag_outputs),
            "custom_metrics": str(args.custom_metrics),
            "ragas_report": str(args.ragas_report),
        },
        "overall": summarize_rows(rows),
        "by_mode": group_rows(rows, "mode"),
        "by_expected_material_type": group_rows(rows, "expected_material_type"),
        "by_eval_type": group_rows(rows, "eval_type"),
        "by_expected_topic": group_rows(rows, "expected_topic"),
        "by_expected_source": group_rows(rows, "expected_source"),
        "custom_metrics": read_json(args.custom_metrics),
        "ragas": summarize_ragas(read_csv(args.ragas_report)),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(build_markdown(summary), encoding="utf-8")

    print(f"Saved JSON summary to {args.output_json}")
    print(f"Saved Markdown summary to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
