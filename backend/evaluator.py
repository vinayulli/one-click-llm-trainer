"""Evaluation orchestrator — launches eval jobs on RunPod."""

from __future__ import annotations

import json

from loguru import logger

from backend.config import Settings
from backend.runpod_client import create_eval_pod
from backend.storage import create_job, get_latest_job, update_job, update_project_stage


async def start_evaluation(project_id: str, settings: Settings) -> dict:
    """Create a RunPod pod to evaluate base vs fine-tuned model."""
    # Get training metadata to find the base model and HF repo
    train_job = await get_latest_job(project_id, "train")
    if not train_job or train_job.status != "completed":
        raise RuntimeError("Training must complete before evaluation.")

    train_meta = json.loads(train_job.metadata_json or "{}")
    base_model = train_meta.get("base_model", "")
    worker_repo = train_meta.get("worker_repo", "")
    hf_username = settings.hf_username
    hf_model_repo = f"{hf_username}/{settings.huggingface.repo_prefix}-{project_id}"

    # If worker scripts weren't uploaded during training, upload now
    if not worker_repo:
        from backend.hf_uploader import upload_worker_scripts
        worker_repo = f"{hf_username}/{settings.huggingface.repo_prefix}-worker"
        upload_worker_scripts(repo_name=worker_repo, hf_token=settings.hf_token)

    job = await create_job(
        project_id=project_id,
        job_type="evaluate",
        metadata={
            "base_model": base_model,
            "finetuned_model": hf_model_repo,
            "worker_repo": worker_repo,
        },
    )

    pod = create_eval_pod(
        settings=settings,
        job_id=job.id,
        project_id=project_id,
        base_model=base_model,
        hf_model_repo=hf_model_repo,
        worker_repo=worker_repo,
    )

    pod_id = pod.get("id", "")
    await update_job(job.id, status="running", metadata={"runpod_pod_id": pod_id})
    await update_project_stage(project_id, "evaluating")

    return {
        "job_id": job.id,
        "runpod_pod_id": pod_id,
        "base_model": base_model,
        "finetuned_model": hf_model_repo,
        "status": "running",
    }


async def get_eval_status(project_id: str, settings: Settings) -> dict:
    """Poll RunPod for eval job status."""
    from backend.runpod_client import get_pod_status

    job = await get_latest_job(project_id, "evaluate")
    if not job:
        return {"status": "no_eval_job", "project_id": project_id}

    metadata = json.loads(job.metadata_json or "{}")
    pod_id = metadata.get("runpod_pod_id", "")

    result = {
        "job_id": job.id,
        "project_id": project_id,
        "status": job.status,
        "base_model": metadata.get("base_model", ""),
    }

    if pod_id and job.status in ("pending", "running"):
        try:
            pod_info = get_pod_status(settings, pod_id)
            pod_status = pod_info.get("desiredStatus", "unknown")

            if pod_status == "EXITED":
                await update_job(job.id, status="completed")
                await update_project_stage(project_id, "evaluation_complete")
                result["status"] = "completed"
            else:
                result["pod_status"] = pod_status
        except Exception as e:
            logger.error(f"Failed to fetch eval pod status: {e}")

    # If completed, try to load results from the eval_results stored in job metadata
    if result["status"] == "completed":
        result["eval_results"] = metadata.get("eval_results", {})

    return result


async def save_eval_results(project_id: str, results: dict):
    """Called by the status poller when eval results are available."""
    job = await get_latest_job(project_id, "evaluate")
    if job:
        await update_job(job.id, metadata={"eval_results": results})
