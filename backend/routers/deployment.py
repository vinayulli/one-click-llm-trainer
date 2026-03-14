import json

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.deployer import (
    deploy_model,
    generate_sample_code,
    get_deployment_status,
    stop_deployment,
)
from backend.models import DeploymentResponse, JobStatus, SampleCodeResponse
from backend.storage import get_latest_job, get_project

router = APIRouter(prefix="/api/projects/{project_id}", tags=["deployment"])


@router.post("/deploy", response_model=DeploymentResponse)
async def deploy(project_id: str, gpu_type: str | None = None):
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = await deploy_model(project_id, settings, gpu_type)
    return DeploymentResponse(
        project_id=project_id,
        status=JobStatus(result.get("status", "pending")),
        runpod_pod_id=result.get("runpod_pod_id", ""),
        hf_repo_url=f"https://huggingface.co/{result.get('hf_model_repo', '')}",
        model_name=result.get("hf_model_repo", ""),
    )


@router.get("/deploy/status", response_model=DeploymentResponse)
async def deployment_status(project_id: str):
    result = await get_deployment_status(project_id, settings)
    return DeploymentResponse(
        project_id=project_id,
        status=JobStatus(result.get("status", "pending")),
        endpoint_url=result.get("endpoint_url", ""),
        runpod_pod_id=result.get("runpod_pod_id", result.get("job_id", "")),
        hf_repo_url=f"https://huggingface.co/{result.get('hf_model_repo', '')}",
        model_name=result.get("hf_model_repo", ""),
    )


@router.delete("/deploy")
async def undeploy(project_id: str):
    return await stop_deployment(project_id, settings)


@router.get("/deploy/sample-code", response_model=SampleCodeResponse)
async def get_usage_code(project_id: str):
    result = await get_deployment_status(project_id, settings)
    endpoint_url = result.get("endpoint_url", "")
    model_name = result.get("hf_model_repo", "")

    if not endpoint_url:
        raise HTTPException(status_code=404, detail="Model not deployed yet")

    code = generate_sample_code(endpoint_url, model_name)
    return SampleCodeResponse(**code)
