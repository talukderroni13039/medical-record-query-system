from typing import Literal

from pydantic import BaseModel


class HealthResult(BaseModel):
    status: Literal["ok"]
    environment: str
    model_loaded: bool
    vector_store: Literal["ok"]
    answer_provider: str
