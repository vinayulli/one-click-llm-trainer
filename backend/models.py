from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"


class ProjectStage(str, Enum):
    CREATED = "created"
    DOCUMENTS_UPLOADED = "documents_uploaded"
    DOCUMENTS_PROCESSED = "documents_processed"
    DATASET_GENERATED = "dataset_generated"
    MODEL_SUGGESTED = "model_suggested"
    TRAINING = "training"
    TRAINING_COMPLETE = "training_complete"
    EVALUATING = "evaluating"
    EVALUATION_COMPLETE = "evaluation_complete"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class TrainRequest(BaseModel):
    base_model: str | None = None  # None = use auto-selected
    gpu_type: str | None = None


class DeployRequest(BaseModel):
    gpu_type: str | None = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    stage: ProjectStage
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    project_id: str
    files_saved: list[str]
    total_files: int


class ProcessResponse(BaseModel):
    project_id: str
    files_processed: list[str]
    total_chunks: int


class DatasetStats(BaseModel):
    total_examples: int
    train_count: int
    validation_count: int
    eval_count: int
    avg_instruction_length: float = 0.0
    avg_output_length: float = 0.0
    estimated_tokens: int = 0


class DatasetResponse(BaseModel):
    project_id: str
    stats: DatasetStats
    preview: list[dict] = []


class ModelSuggestion(BaseModel):
    model_id: str
    display_name: str
    parameter_count: str
    score: float
    reasoning: str
    recommended_gpu: str
    estimated_train_time_hours: float
    estimated_cost_usd: float


class ModelSuggestionResponse(BaseModel):
    project_id: str
    suggestions: list[ModelSuggestion]
    auto_selected: str  # model_id of top pick


class TrainingStatusResponse(BaseModel):
    project_id: str
    job_id: str
    status: JobStatus
    base_model: str = ""
    runpod_pod_id: str = ""
    progress: dict = Field(default_factory=dict)
    # progress includes: step, total_steps, loss, epoch, eta_seconds


class EvalResultsResponse(BaseModel):
    project_id: str
    status: JobStatus
    base_model: str = ""
    num_examples: int = 0
    base_avg_metrics: dict = Field(default_factory=dict)
    finetuned_avg_metrics: dict = Field(default_factory=dict)
    details: list[dict] = []


class DeploymentResponse(BaseModel):
    project_id: str
    status: JobStatus
    endpoint_url: str = ""
    runpod_pod_id: str = ""
    hf_repo_url: str = ""
    model_name: str = ""


class SampleCodeResponse(BaseModel):
    python_openai: str
    python_requests: str
    curl: str
    javascript: str
