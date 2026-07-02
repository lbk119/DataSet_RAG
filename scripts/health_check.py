from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENV = [
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "LLM_DEFAULT_MODEL",
    "VL_MODEL",
    "MILVUS_URL",
    "CHUNKS_COLLECTION",
    "MONGO_URL",
    "MONGO_DB_NAME",
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_BUCKET_NAME",
]


def parse_host_port(value: str | None, default_port: int | None = None) -> tuple[str, int] | None:
    if not value:
        return None
    raw = value.strip()
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    raw = raw.split("/", 1)[0]
    if ":" in raw:
        host, port = raw.rsplit(":", 1)
        return host or "127.0.0.1", int(port)
    if default_port is None:
        return None
    return raw, default_port


def check_tcp(name: str, target: tuple[str, int] | None, timeout: float = 1.5) -> bool:
    if not target:
        print(f"[WARN] {name}: skipped, no host/port configured")
        return False
    host, port = target
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print(f"[OK]   {name}: {host}:{port}")
            return True
    except OSError as exc:
        print(f"[FAIL] {name}: {host}:{port} ({exc})")
        return False


def check_http(name: str, url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            if 200 <= response.status < 300:
                print(f"[OK]   {name}: {url}")
                return True
            print(f"[FAIL] {name}: {url} status={response.status}")
            return False
    except Exception as exc:
        print(f"[WARN] {name}: {url} ({exc})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local DataSet_RAG development environment.")
    parser.add_argument("--services", action="store_true", help="Also probe local FastAPI service health endpoints.")
    args = parser.parse_args()

    env_path = ROOT / ".env"
    if not env_path.exists():
        print("[FAIL] .env not found. Copy .env.example to .env first.")
        return 1

    load_dotenv(env_path)
    print(f"[OK]   loaded {env_path}")

    missing = [key for key in REQUIRED_ENV if not os.getenv(key)]
    if missing:
        print("[FAIL] Missing required env vars: " + ", ".join(missing))
    else:
        print("[OK]   required env vars present")

    all_ok = not missing
    all_ok &= check_tcp("Milvus", parse_host_port(os.getenv("MILVUS_URL"), 19530))
    all_ok &= check_tcp("MongoDB", parse_host_port(os.getenv("MONGO_URL"), 27017))
    all_ok &= check_tcp("MinIO", parse_host_port(os.getenv("MINIO_ENDPOINT"), 9000))

    if args.services:
        all_ok &= check_http("Import API", "http://127.0.0.1:8001/courses")
        all_ok &= check_http("Query API", "http://127.0.0.1:8002/health")

    if all_ok:
        print("[OK]   environment check passed")
        return 0
    print("[WARN] environment check finished with issues")
    return 2


if __name__ == "__main__":
    sys.exit(main())
