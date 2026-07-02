from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from copy import deepcopy

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import _answer_relevancy, _context_precision, _context_recall, _faithfulness


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "evals" / "reports" / "rag_outputs.jsonl"
DEFAULT_OUTPUT = ROOT / "evals" / "reports" / "ragas_report.csv"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row.get("answer", ""),
                "retrieved_contexts": row.get("contexts", []),
                "reference": row.get("reference", ""),
                "id": row.get("id", ""),
            }
            for row in rows
            if row.get("answer") and row.get("contexts") and not row.get("error")
        ]
    )


def tokenize_for_relevancy(text: str) -> set[str]:
    text = (text or "").lower()
    chinese_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
    words = set(re.findall(r"[a-z0-9_]{2,}", text))
    math_terms = set(re.findall(r"[a-z]\d*|\\[a-z]+", text))
    return {token for token in chinese_chars | words | math_terms if token.strip()}


def lexical_answer_relevancy(question: str, answer: str) -> float:
    q_tokens = tokenize_for_relevancy(question)
    a_tokens = tokenize_for_relevancy(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    recall = len(q_tokens & a_tokens) / len(q_tokens)
    precision = len(q_tokens & a_tokens) / len(a_tokens)
    if recall + precision == 0:
        return 0.0
    return round((2 * recall * precision) / (recall + precision), 4)


def _is_missing_metric(value: Any) -> bool:
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return math.isnan(number) or math.isinf(number)


def fill_answer_relevancy_fallback(frame: pd.DataFrame) -> pd.DataFrame:
    fallback_values = [
        lexical_answer_relevancy(question, answer)
        for question, answer in zip(frame.get("user_input", []), frame.get("response", []))
    ]
    frame["answer_relevancy_fallback"] = fallback_values
    if "answer_relevancy" not in frame:
        frame["answer_relevancy"] = fallback_values
        frame["answer_relevancy_note"] = "fallback_lexical_overlap; check RAGAS_EMBEDDING_MODEL compatibility"
        return frame
    missing_mask = frame["answer_relevancy"].map(_is_missing_metric)
    if missing_mask.any():
        frame.loc[missing_mask, "answer_relevancy"] = frame.loc[missing_mask, "answer_relevancy_fallback"]
        frame["answer_relevancy_note"] = ""
        frame.loc[missing_mask, "answer_relevancy_note"] = "fallback_lexical_overlap; check RAGAS_EMBEDDING_MODEL compatibility"
    return frame


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Evaluate RAG outputs with Ragas.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("RAGAS_LLM_MODEL") or os.getenv("LLM_DEFAULT_MODEL"))
    parser.add_argument("--embedding-model", default=os.getenv("RAGAS_EMBEDDING_MODEL") or "text-embedding-v4")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N evaluable rows.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    dataset = build_dataset(rows)
    if len(dataset) == 0:
        print("No evaluable rows. Run evals/run_rag_eval.py first and ensure answers/contexts are present.")
        return 1

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        print("OPENAI_API_KEY / OPENAI_BASE_URL are required for Ragas LLM judging.")
        return 1
    if not args.model:
        print("RAGAS_LLM_MODEL or LLM_DEFAULT_MODEL is required. You can also pass --model explicitly.")
        return 1
    if not args.embedding_model:
        print("RAGAS_EMBEDDING_MODEL is required. You can also pass --embedding-model explicitly.")
        return 1

    llm = ChatOpenAI(
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        n=1,
        extra_body={"enable_thinking": False},
        disabled_params={"parallel_tool_calls": None},
    )
    embeddings = OpenAIEmbeddings(
        model=args.embedding_model,
        api_key=api_key,
        base_url=base_url,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    try:
        probe = embeddings.embed_query("Ragas embedding compatibility probe")
        if not isinstance(probe, list) or not probe:
            raise ValueError("embedding response is empty")
        print(f"Embedding probe OK: model={args.embedding_model}, dim={len(probe)}")
    except Exception as exc:
        print(f"Embedding probe failed for {args.embedding_model}: {exc}")
        print("Ragas answer_relevancy may become NaN. The report will use lexical fallback if needed.")
    answer_relevancy = deepcopy(_answer_relevancy)
    answer_relevancy.strictness = 1
    metrics = [_faithfulness, answer_relevancy, _context_precision, _context_recall]
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings, raise_exceptions=False)

    frame = result.to_pandas()
    frame = fill_answer_relevancy_fallback(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result)
    print(f"Saved Ragas report to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
