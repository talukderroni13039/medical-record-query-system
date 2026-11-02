"""HTTP layer only: receive → call service → wrap in ApiResponse. No business logic here."""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Request

from src.core.dependencies import get_ingestion_service, get_query_service
from src.core.schemas import ApiResponse, ErrorResponse, ResponseMeta
from src.domains.patient_records.schemas import (
    IngestionResult,
    IngestRequest,
    QueryRequest,
    QueryResult,
    StatsResult,
)
from src.domains.patient_records.services.ingestion_service import IngestionService
from src.domains.patient_records.services.query_service import QueryService

router = APIRouter(prefix="/patient-records", tags=["patient-records"])


def _errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {code: {"model": ErrorResponse} for code in (*codes, 422, 500)}


@router.post(
    "/ingest",
    response_model=ApiResponse[IngestionResult],
    summary="Index records from RECORDS_DIR (idempotent)",
    responses=_errors(409, 503),
)
def ingest(
    request: Request,
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    body: Annotated[IngestRequest | None, Body()] = None,
) -> ApiResponse[IngestionResult]:
    # Plain `def`: FastAPI runs it in a threadpool, so CPU-bound embedding never blocks the loop.
    result = service.ingest(force_reingest=(body or IngestRequest()).force_reingest)
    return ApiResponse(data=result, meta=ResponseMeta.from_request(request))


@router.post(
    "/query",
    response_model=ApiResponse[QueryResult],
    summary="Semantic search over indexed records (retrieval only, not medical advice)",
    responses=_errors(400, 409, 502, 503, 504),
)
async def query(
    request: Request,
    body: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service)],
) -> ApiResponse[QueryResult]:
    result = await service.query(body)
    return ApiResponse(data=result, meta=ResponseMeta.from_request(request))


@router.get(
    "/stats",
    response_model=ApiResponse[StatsResult],
    summary="Index statistics",
    responses=_errors(503),
)
def stats(
    request: Request,
    service: Annotated[QueryService, Depends(get_query_service)],
) -> ApiResponse[StatsResult]:
    return ApiResponse(data=service.stats(), meta=ResponseMeta.from_request(request))
