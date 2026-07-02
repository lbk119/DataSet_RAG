from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC_DIR = ROOT / "evals" / "sample_docs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload sample evaluation PDFs to the import service.")
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--doc-dir", type=Path, default=DEFAULT_DOC_DIR)
    parser.add_argument("--course-name", default="计算方法")
    args = parser.parse_args()

    pdfs = sorted(args.doc_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.doc_dir}")
        return 1

    files = []
    opened = []
    try:
        for path in pdfs:
            handle = path.open("rb")
            opened.append(handle)
            files.append(("files", (path.name, handle, "application/pdf")))

        data = {"course_name": args.course_name, "material_type": "exam"}
        print(f"Uploading {len(files)} files to {args.api}/upload ...")
        response = requests.post(f"{args.api}/upload", data=data, files=files, timeout=120)
        response.raise_for_status()
        payload = response.json()
        print(payload)

        task_ids = payload.get("task_ids", [])
        if not task_ids:
            return 0

        print("Polling import tasks ...")
        unfinished = set(task_ids)
        while unfinished:
            for task_id in list(unfinished):
                status_response = requests.get(f"{args.api}/status/{task_id}", timeout=10)
                status_response.raise_for_status()
                status = status_response.json().get("status", "")
                print(f"{task_id}: {status}")
                if status in {"completed", "failed"}:
                    unfinished.remove(task_id)
            if unfinished:
                time.sleep(3)
        return 0
    finally:
        for handle in opened:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
