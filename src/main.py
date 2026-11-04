"""Application entry point.

Environment loading MUST happen before any other `src` import, because modules read
`os.getenv(...)` at import or construction time.
"""

import os

from dotenv import load_dotenv

APP_ENV = os.getenv("APP_ENV", "dev")
load_dotenv(f".env.{APP_ENV}")  # never overrides variables already set (shell / Docker)

# Chroma telemetry off even if a client is created without our Settings.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import logging  # noqa: E402
from typing import Annotated  # noqa: E402

from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402

from src.api_router import api_router  # noqa: E402
from src.core.dependencies import get_embedder, get_vector_store  # noqa: E402
from src.core.exception_handlers import global_exception_handler  # noqa: E402
from src.core.exceptions import AppError  # noqa: E402
from src.core.lifespan import answer_provider, lifespan  # noqa: E402
from src.core.middleware import RequestContextMiddleware  # noqa: E402
from src.core.schemas import ApiResponse, ErrorResponse, HealthResult, ResponseMeta  # noqa: E402
from src.embeddings.base import Embedder  # noqa: E402
from src.vectorstore.base import VectorStore  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local Medical Record Query API",
        version="0.1.0",
        description=(
            "On-premise semantic search over **synthetic** patient records.\n\n"
            "Retrieval only by default (`ANSWER_PROVIDER=none`). "
            "**Not medical advice**: results are record excerpts, not diagnoses or treatment."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)
    # One handler function, registered three times (see core/exception_handlers.py).
    app.add_exception_handler(AppError, global_exception_handler)
    app.add_exception_handler(RequestValidationError, global_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    app.include_router(api_router)

    @app.get(
        "/health",
        tags=["health"],
        response_model=ApiResponse[HealthResult],
        responses={503: {"model": ErrorResponse}},
    )
    def health(
        request: Request,
        embedder: Annotated[Embedder, Depends(get_embedder)],  # raises 503 MODEL_NOT_READY
        store: Annotated[VectorStore, Depends(get_vector_store)],  # raises 503 VECTOR_STORE_ERROR
    ) -> ApiResponse[HealthResult]:
        store.ping()
        result = HealthResult(
            status="ok",
            environment=APP_ENV,
            model_loaded=embedder is not None,
            vector_store="ok",
            answer_provider=answer_provider(),
        )
        return ApiResponse(data=result, meta=ResponseMeta.from_request(request))

    return app


app = create_app()
