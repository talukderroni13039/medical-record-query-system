"""Provider-neutral answer-generation schemas.

`GeneratedAnswer` is the Instructor `response_model` (validated output from the LLM) and also the
`answer` field of `QueryResult`.
"""

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

ALLOWED_IDS_CONTEXT_KEY = "allowed_chunk_ids"


class ContextChunk(BaseModel):
    """One retrieved chunk handed to the answer generator. Only this text leaves the process."""

    chunk_id: str
    patient_id: str
    section: str
    text: str


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        min_length=1,
        description=(
            "Answer using ONLY facts stated in the provided chunks, citing chunk IDs in square "
            "brackets, e.g. [P-0007.md:4]. If the chunks do not contain the answer, say so plainly."
        ),
    )
    cited_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Exact chunk IDs (from the provided chunks) that support the answer.",
    )
    insufficient_context: bool = Field(
        description="True if the provided chunks do not contain enough information to answer."
    )

    @field_validator("cited_chunk_ids")
    @classmethod
    def cited_ids_must_be_retrieved(cls, value: list[str], info: ValidationInfo) -> list[str]:
        deduped = list(dict.fromkeys(v.strip() for v in value if v.strip()))
        allowed = (info.context or {}).get(ALLOWED_IDS_CONTEXT_KEY)
        if allowed is not None:
            unknown = [cid for cid in deduped if cid not in allowed]
            if unknown:
                raise ValueError(
                    f"cited_chunk_ids contains IDs that were not provided: {unknown}. "
                    "Cite only chunk IDs from the provided chunks."
                )
        return deduped

    @model_validator(mode="after")
    def answers_need_citations(self) -> "GeneratedAnswer":
        if not self.insufficient_context and not self.cited_chunk_ids:
            raise ValueError(
                "An answer that is not marked insufficient_context must cite at least one chunk."
            )
        return self
