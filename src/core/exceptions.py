"""Application errors. Each carries its HTTP status and a stable machine-readable code."""


class AppError(Exception):
    status_code: int = 500
    code: str = "APP_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidInputError(AppError):
    status_code = 400
    code = "INVALID_INPUT"


class AnswerGenerationDisabledError(AppError):
    status_code = 400
    code = "ANSWER_GENERATION_DISABLED"


class IngestionInProgressError(AppError):
    status_code = 409
    code = "INGESTION_IN_PROGRESS"


class CollectionEmptyError(AppError):
    status_code = 409
    code = "COLLECTION_EMPTY"


class RecordsDirectoryError(AppError):
    status_code = 500
    code = "RECORDS_DIR_ERROR"


class AnswerGenerationError(AppError):
    status_code = 502
    code = "ANSWER_GENERATION_FAILED"


class ModelNotReadyError(AppError):
    status_code = 503
    code = "MODEL_NOT_READY"


class VectorStoreError(AppError):
    status_code = 503
    code = "VECTOR_STORE_ERROR"


class LLMTimeoutError(AppError):
    status_code = 504
    code = "LLM_TIMEOUT"
