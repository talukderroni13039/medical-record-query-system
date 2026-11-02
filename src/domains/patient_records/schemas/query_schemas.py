import os
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]


class QueryFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str | None = Field(default=None, max_length=64, examples=["P-0007"])
    section: str | None = Field(
        default=None,
        max_length=64,
        description="One of the known record sections (case-insensitive), e.g. Allergies.",
        examples=["Allergies"],
    )


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "question": "Which patients are allergic to penicillin?",
                    "top_k": 5,
                    "filters": {"patient_id": None, "section": "Allergies"},
                }
            ]
        },
    )

    question: Question
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    filters: QueryFilters = Field(default_factory=QueryFilters)


class ChunkSource(BaseModel):
    patient_id: str
    file_name: str
    section: str
    record_date: str
    chunk_index: int


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(description='Stable ID "{file_name}:{chunk_index}".')
    text: str
    score: float = Field(description="Cosine similarity (1 - cosine distance). Higher is closer.")
    source: ChunkSource


class QueryResult(BaseModel):
    question: str
    results: list[RetrievedChunk]
