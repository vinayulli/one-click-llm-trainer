"""Training orchestrator — launches training jobs on RunPod."""

from __future__ import annotations

import json

from loguru import logger

from backend.config import Settings
from backend.hf_uploader import upload_dataset_to_hub, upload_worker_scripts
from backend.runpod_client import create_training_pod
from backend.storage import create_job, update_job, update_project_stage


async def start_training(
    project_id: str,
    base_model: str,
    settings: Settings,
    gpu_type: str | None = None,
) -> dict:
    """
    1. Upload worker scripts to HF (so pods can pull them — no Docker build)
    2. Upload dataset to HF
    3. Create a RunPod GPU pod using standard base image
    4. Pod startup script installs deps, pulls code from HF, runs training
    """
    dataset_dir = settings.project_dataset_dir(project_id)
    train_path = dataset_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError("Training dataset not found. Generate the dataset first.")

    hf_username = settings.hf_username
    hf_token = settings.hf_token
    repo_prefix = settings.huggingface.repo_prefix

    # 1) Upload worker scripts to HF (reusable across projects)
    worker_repo = f"{hf_username}/{repo_prefix}-worker"
    logger.info(f"Uploading worker scripts to HF: {worker_repo}")
    upload_worker_scripts(repo_name=worker_repo, hf_token=hf_token)

    # 2) Upload dataset to HF so the worker can pull it
    dataset_repo = f"{hf_username}/{repo_prefix}-data-{project_id}"
    logger.info(f"Uploading dataset to HF: {dataset_repo}")
    upload_dataset_to_hub(
        local_path=str(dataset_dir),
        repo_name=dataset_repo,
        hf_token=hf_token,
        private=True,
    )

    # 3) Create job record
    job = await create_job(
        project_id=project_id,
        job_type="train",
        metadata={
            "base_model": base_model,
            "dataset_repo": dataset_repo,
            "worker_repo": worker_repo,
            "gpu_type": gpu_type or settings.runpod.gpu_type_id,
        },
    )

    # 4) Create RunPod pod (standard base image, no Docker build)
    pod = create_training_pod(
        settings=settings,
        job_id=job.id,
        project_id=project_id,
        base_model=base_model,
        worker_repo=worker_repo,
        gpu_type=gpu_type,
    )

    pod_id = pod.get("id", "")
    await update_job(job.id, status="running", metadata={"runpod_pod_id": pod_id})
    await update_project_stage(project_id, "training")

    return {
        "job_id": job.id,
        "runpod_pod_id": pod_id,
        "base_model": base_model,
        "dataset_repo": dataset_repo,
        "status": "running",
    }


async def get_training_status(project_id: str, settings: Settings) -> dict:
    """Poll RunPod for training job status."""
    from backend.runpod_client import get_pod_status
    from backend.storage import get_latest_job

    job = await get_latest_job(project_id, "train")
    if not job:
        return {"status": "no_training_job", "project_id": project_id}

    metadata = json.loads(job.metadata_json or "{}")
    pod_id = metadata.get("runpod_pod_id", "")

    result = {
        "job_id": job.id,
        "project_id": project_id,
        "status": job.status,
        "base_model": metadata.get("base_model", ""),
        "runpod_pod_id": pod_id,
        "progress": {},
    }

    if pod_id and job.status in ("pending", "running"):
        try:
            pod_info = get_pod_status(settings, pod_id)
            pod_status = pod_info.get("desiredStatus", "unknown")
            runtime = pod_info.get("runtime", {})

            result["pod_status"] = pod_status
            result["progress"] = {
                "uptime_seconds": runtime.get("uptimeInSeconds", 0) if runtime else 0,
                "gpu_util": runtime.get("gpus", [{}])[0].get("gpuUtilPerc", 0) if runtime and runtime.get("gpus") else 0,
            }

            # If pod has exited, mark job complete or failed
            if pod_status == "EXITED":
                await update_job(job.id, status="completed")
                await update_project_stage(project_id, "training_complete")
                result["status"] = "completed"

        except Exception as e:
            logger.error(f"Failed to fetch pod status: {e}")
            result["pod_status_error"] = str(e)

    return result


async def cancel_training(project_id: str, settings: Settings) -> dict:
    """Cancel a running training job by terminating the RunPod pod."""
    from backend.runpod_client import terminate_pod
    from backend.storage import get_latest_job

    job = await get_latest_job(project_id, "train")
    if not job:
        return {"status": "no_training_job"}

    metadata = json.loads(job.metadata_json or "{}")
    pod_id = metadata.get("runpod_pod_id", "")

    if pod_id:
        terminate_pod(settings, pod_id)

    await update_job(job.id, status="cancelled")
    await update_project_stage(project_id, "dataset_generated")

    return {"status": "cancelled", "job_id": job.id}
