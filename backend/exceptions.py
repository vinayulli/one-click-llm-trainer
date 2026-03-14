from __future__ import annotations

from fastapi import HTTPException, status


class OCLTError(Exception):
    """Base exception for One Click LLM Trainer."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ProjectNotFoundError(OCLTError):
    def __init__(self, project_id: str):
        super().__init__(
            f"Project '{project_id}' not found", status.HTTP_404_NOT_FOUND
        )


class StageError(OCLTError):
    """Raised when an action is attempted at the wrong pipeline stage."""

    def __init__(self, required: str, current: str):
        super().__init__(
            f"This action requires stage '{required}', but project is at '{current}'",
            status.HTTP_409_CONFLICT,
        )


class RunPodError(OCLTError):
    def __init__(self, message: str):
        super().__init__(f"RunPod error: {message}", status.HTTP_502_BAD_GATEWAY)


class DatasetError(OCLTError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)


class TrainingError(OCLTError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


class DeploymentError(OCLTError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


def oclt_exception_handler(exc: OCLTError):
    raise HTTPException(status_code=exc.status_code, detail=exc.message)
