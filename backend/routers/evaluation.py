from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.evaluator import get_eval_status, start_evaluation
from backend.models import EvalResultsResponse, JobStatus
from backend.storage import get_project

router = APIRouter(prefix="/api/projects/{project_id}", tags=["evaluation"])


@router.post("/evaluate")
async def evaluate_model(project_id: str):
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = await start_evaluation(project_id, settings)
    return result


@router.get("/evaluate/status")
async def evaluation_status(project_id: str):
    return await get_eval_status(project_id, settings)


@router.get("/evaluate/results", response_model=EvalResultsResponse)
async def evaluation_results(project_id: str):
    result = await get_eval_status(project_id, settings)

    eval_results = result.get("eval_results", {})
    return EvalResultsResponse(
        project_id=project_id,
        status=JobStatus(result.get("status", "pending")),
        base_model=result.get("base_model", ""),
        num_examples=eval_results.get("num_examples", 0),
        base_avg_metrics=eval_results.get("base_avg", {}),
        finetuned_avg_metrics=eval_results.get("finetuned_avg", {}),
        details=eval_results.get("details", []),
    )
