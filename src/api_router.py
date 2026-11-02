"""Central /api/v1 router. Every domain router is included here."""

from fastapi import APIRouter

from src.domains.patient_records.routers.api_endpoints import router as patient_records_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(patient_records_router)
