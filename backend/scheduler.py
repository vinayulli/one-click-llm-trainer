"""Background scheduler that polls RunPod for active job statuses."""

from __future__ import annotations

import json

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from backend.config import settings
from backend.runpod_client import get_pod_status
from backend.storage import get_active_jobs, update_job, update_project_stage

scheduler = AsyncIOScheduler()


async def poll_runpod_jobs():
    """Poll all active jobs for status updates."""
    jobs = await get_active_jobs()
    if not jobs:
        return

    for job in jobs:
        metadata = json.loads(job.metadata_json or "{}")
        pod_id = metadata.get("runpod_pod_id", "")

        if not pod_id:
            continue

        try:
            pod_info = get_pod_status(settings, pod_id)
            pod_status = pod_info.get("desiredStatus", "unknown")
            runtime = pod_info.get("runtime", {})

            progress = {}
            if runtime:
                progress["uptime_seconds"] = runtime.get("uptimeInSeconds", 0)
                gpus = runtime.get("gpus", [])
                if gpus:
                    progress["gpu_util"] = gpus[0].get("gpuUtilPerc", 0)

            if pod_status == "EXITED":
                new_stage = {
                    "train": "training_complete",
                    "evaluate": "evaluation_complete",
                    "deploy": "deployed",
                }.get(job.job_type, job.job_type + "_complete")

                await update_job(job.id, status="completed", metadata={"progress": progress})
                await update_project_stage(job.project_id, new_stage)
                logger.info(f"Job {job.id} ({job.job_type}) completed for project {job.project_id}")

            elif pod_status == "RUNNING":
                await update_job(job.id, status="running", metadata={"progress": progress})

            elif pod_status in ("ERROR", "TERMINATED"):
                await update_job(job.id, status="failed", metadata={"error": f"Pod status: {pod_status}"})
                logger.error(f"Job {job.id} failed: pod status {pod_status}")

        except Exception as e:
            logger.error(f"Error polling job {job.id}: {e}")


def start_scheduler():
    scheduler.add_job(poll_runpod_jobs, "interval", seconds=30, id="poll_runpod")
    scheduler.start()
    logger.info("Background scheduler started (polling every 30s)")


def stop_scheduler():
    scheduler.shutdown()
