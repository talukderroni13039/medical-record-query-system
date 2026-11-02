"""Shared response envelope used by every endpoint and every error."""

import time

from fastapi import Request
from pydantic import BaseModel


class ResponseMeta(BaseModel):
    request_id: str | None = None
    processing_time_ms: int | None = None

    @classmethod
    def from_request(cls, request: Request) -> "ResponseMeta":
        start: float | None = getattr(request.state, "start_time", None)
        return cls(
            request_id=getattr(request.state, "request_id", None),
            processing_time_ms=int((time.perf_counter() - start) * 1000) if start else None,
        )


class ErrorDetail(BaseModel):
    code: str
    message: str


class ApiResponse[T](BaseModel):
    success: bool = True
    data: T
    meta: ResponseMeta


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
    meta: ResponseMeta
