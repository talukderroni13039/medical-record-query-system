from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    force_reingest: bool = Field(
        default=False, description="Re-embed every file even if its content hash is unchanged."
    )


class RejectedFile(BaseModel):
    file_name: str
    reason: str = Field(examples=["PII detected: email"])


class IngestionResult(BaseModel):
    files_scanned: int
    files_ingested: int
    files_skipped_unchanged: int
    files_rejected: list[RejectedFile]
    chunks_upserted: int


class StatsResult(BaseModel):
    collection: str
    chunk_count: int
    file_count: int
    embedding_model: str
    dimensions: int
