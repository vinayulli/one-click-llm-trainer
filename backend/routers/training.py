from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.models import TrainRequest, TrainingStatusResponse, JobStatus
from backend.storage import get_project
from backend.trainer import cancel_training, get_training_status, start_training

router = APIRouter(prefix="/api/projects/{project_id}", tags=["training"])


@router.post("/train")
async def train_model(project_id: str, req: TrainRequest | None = None):
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    base_model = req.base_model if req else None
    gpu_type = req.gpu_type if req else None

    # If no model specified, auto-select by calling suggest endpoint logic
    if not base_model:
        from backend.routers.datasets import get_model_suggestions
        suggestions = await get_model_suggestions(project_id)
        base_model = suggestions.auto_selected

    result = await start_training(
        project_id=project_id,
        base_model=base_model,
        settings=settings,
        gpu_type=gpu_type,
    )
    return result


@router.get("/train/status", response_model=TrainingStatusResponse)
async def train_status(project_id: str):
    result = await get_training_status(project_id, settings)
    return TrainingStatusResponse(
        project_id=project_id,
        job_id=result.get("job_id", ""),
        status=JobStatus(result.get("status", "pending")),
        base_model=result.get("base_model", ""),
        runpod_pod_id=result.get("runpod_pod_id", ""),
        progress=result.get("progress", {}),
    )


@router.post("/train/cancel")
async def cancel_train(project_id: str):
    return await cancel_training(project_id, settings)
