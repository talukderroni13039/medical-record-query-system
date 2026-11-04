"""CLI ingest: same IngestionService as the API, same construction path (core/lifespan.py).

Usage (PowerShell):
    uv run python scripts/ingest.py            # incremental (unchanged files skipped)
    uv run python scripts/ingest.py --force    # re-embed everything
"""

import argparse
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(f".env.{os.getenv('APP_ENV', 'dev')}")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest patient records into the vector index.")
    parser.add_argument("--force", action="store_true", help="re-embed unchanged files too")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")

    from src.core.dependencies import get_ingestion_service, get_records_dir
    from src.core.exceptions import AppError
    from src.core.lifespan import build_embedder, build_vector_store

    try:
        service = get_ingestion_service(
            embedder=build_embedder(),
            store=build_vector_store(),
            lock=threading.Lock(),
            records_dir=get_records_dir(),
        )
        result = service.ingest(force_reingest=args.force)
    except AppError as exc:
        print(f"ERROR {exc.code}: {exc.message}", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
