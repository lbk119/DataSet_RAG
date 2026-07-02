from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clients.course_utils import ensure_course
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.utils.task_utils import update_task_status


DEFAULT_DATASET = ROOT / "evals" / "datasets" / "calculation_methods_eval.jsonl"
DEFAULT_OUTPUT = ROOT / "evals" / "reports" / "rag_outputs.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _timed_node(name: str, state: dict[str, Any], fn) -> dict[str, Any]:
    started = time.perf_counter()
    result = fn(state) or {}
    elapsed = time.perf_counter() - started
    print(f"    {name}: {elapsed:.2f}s")
    return result


def run_single_case(case: dict[str, Any], *, skip_hyde: bool = False) -> dict[str, Any]:
    course = ensure_course(course_id=case.get("course_id"), course_name=case.get("course_name"))
    session_id = f"eval-{case.get('id', uuid.uuid4().hex)}-{uuid.uuid4().hex[:8]}"
    mode = case.get("mode") or "qa"
    started = time.perf_counter()
    state: dict[str, Any] = {
        "original_query": case["question"],
        "session_id": session_id,
        "is_stream": False,
        "course_id": course["course_id"],
        "course_name": course["course_name"],
        "mode": mode,
        "attachment_context": "",
    }
    update_task_status(session_id, "processing", False)
    error = ""
    try:
        state.update(_timed_node("node_item_name_confirm", state, node_item_name_confirm))
        if not state.get("answer"):
            state.update(_timed_node("node_search_embedding", state, node_search_embedding))
            if not skip_hyde:
                state.update(_timed_node("node_search_embedding_hyde", state, node_search_embedding_hyde))
            state.update(_timed_node("node_web_search_mcp", state, node_web_search_mcp))
            state.update(_timed_node("node_rrf", state, node_rrf))
            state.update(_timed_node("node_rerank", state, node_rerank))
        state.update(_timed_node("node_answer_output", state, node_answer_output))
        update_task_status(session_id, "completed", False)
    except Exception as exc:
        error = str(exc)
        update_task_status(session_id, "failed", False)

    latency = time.perf_counter() - started
    reranked_docs = state.get("reranked_docs", []) or []
    contexts = [doc.get("text", "") for doc in reranked_docs if doc.get("text")]
    return {
        "id": case.get("id"),
        "course_id": course["course_id"],
        "course_name": course["course_name"],
        "mode": mode,
        "question": case["question"],
        "reference": case.get("reference", ""),
        "expected_material_type": case.get("expected_material_type", ""),
        "expected_source": case.get("expected_source", ""),
        "expected_topic": case.get("expected_topic", ""),
        "eval_type": case.get("eval_type", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "query_intent": state.get("query_intent", ""),
        "preferred_material_types": state.get("preferred_material_types", []),
        "answer": state.get("answer", ""),
        "contexts": contexts,
        "reranked_docs": reranked_docs,
        "latency_seconds": round(latency, 3),
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DataSet_RAG cases and store RAG outputs for evaluation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-hyde", action="store_true", help="Skip HyDE during quick retrieval smoke tests.")
    args = parser.parse_args()

    cases = read_jsonl(args.dataset)
    if args.limit > 0:
        cases = cases[: args.limit]

    outputs = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.get('id')} {case.get('question')}")
        result = run_single_case(case, skip_hyde=args.skip_hyde)
        print(f"  latency={result['latency_seconds']}s error={result['error'] or '-'} contexts={len(result['contexts'])}")
        outputs.append(result)

    write_jsonl(args.output, outputs)
    print(f"Saved {len(outputs)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
