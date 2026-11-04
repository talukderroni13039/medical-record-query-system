"""The single global exception handler.

Registered for `AppError`, `RequestValidationError`, and `Exception`, so every error, including
FastAPI's own validation errors, is rendered as `ErrorResponse`.
"""

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.core.exceptions import AppError
from src.core.schemas import ErrorDetail, ErrorResponse, ResponseMeta

logger = logging.getLogger(__name__)


def _format_validation_error(exc: RequestValidationError) -> str:
    # Field locations + messages only; never echo input values (they may contain record text).
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        parts.append(f"{loc or 'body'}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or "Invalid request"


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    meta = ResponseMeta.from_request(request)

    if isinstance(exc, AppError):
        logger.warning("%s: %s [request_id=%s]", exc.code, exc.message, meta.request_id)
        status_code, code, message = exc.status_code, exc.code, exc.message
    elif isinstance(exc, RequestValidationError):
        message = _format_validation_error(exc)
        logger.info("VALIDATION_ERROR [request_id=%s]", meta.request_id)
        status_code, code = 422, "VALIDATION_ERROR"
    else:
        logger.exception("Unhandled error [request_id=%s]", meta.request_id)
        status_code, code, message = 500, "INTERNAL_ERROR", "Unexpected server error"

    body = ErrorResponse(error=ErrorDetail(code=code, message=message), meta=meta)
    response = JSONResponse(status_code=status_code, content=body.model_dump())
    if meta.request_id:
        # Unhandled exceptions bypass the middleware's response path, so set the header here too.
        response.headers["X-Request-ID"] = meta.request_id
    return response
