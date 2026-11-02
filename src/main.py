"""Application entry point.

Environment loading MUST happen before any other `src` import, because modules read
`os.getenv(...)` at import or construction time.
"""

import os

from dotenv import load_dotenv

APP_ENV = os.getenv("APP_ENV", "dev")
load_dotenv(f".env.{APP_ENV}")  # never overrides variables already set (shell / Docker)

import logging  # noqa: E402

from fastapi import FastAPI  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local Medical Record Query API",
        version="0.1.0",
        description=(
            "On-premise semantic search over synthetic patient records. "
            "Retrieval only by default; **not medical advice**."
        ),
    )

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "environment": APP_ENV}

    return app


app = create_app()
