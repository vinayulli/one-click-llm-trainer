"""
RunPod SDK wrapper for pod lifecycle management.

Uses standard RunPod base images (runpod/pytorch, vllm/vllm-openai).
Worker code is pulled from HuggingFace at pod startup — no custom Docker build needed.

Startup scripts are passed as base64-encoded env vars to avoid GraphQL
string escaping issues with RunPod's API.
"""

from __future__ import annotations

import base64

import runpod
from loguru import logger

from backend.config import Settings


def _init_runpod(settings: Settings):
    runpod.api_key = settings.runpod_api_key


def _encode_script(script: str) -> str:
    """Base64-encode a startup script for safe transport via env var."""
    return base64.b64encode(script.encode("utf-8")).decode("ascii")


def _training_script(worker_repo: str, hf_token: str) -> str:
    return f"""#!/bin/bash
set -e
echo "=== OCLT: Installing dependencies ==="
pip install --no-cache-dir transformers==4.47.1 peft==0.14.0 trl==0.13.0 accelerate==1.2.1 bitsandbytes==0.45.0 datasets==3.2.0 huggingface-hub loguru==0.7.3 openai==1.58.1
echo "=== OCLT: Downloading worker scripts ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='{worker_repo}', local_dir='/workspace/oclt', token='{hf_token}')"
echo "=== OCLT: Starting training ==="
cd /workspace/oclt
PYTHONPATH=/workspace/oclt python -m worker.train
"""


def _eval_script(worker_repo: str, hf_token: str) -> str:
    return f"""#!/bin/bash
set -e
echo "=== OCLT: Installing dependencies ==="
pip install --no-cache-dir transformers==4.47.1 peft==0.14.0 accelerate==1.2.1 bitsandbytes==0.45.0 datasets==3.2.0 huggingface-hub loguru==0.7.3 openai==1.58.1 rouge-score==0.1.2 scikit-learn==1.6.1
echo "=== OCLT: Downloading worker scripts ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='{worker_repo}', local_dir='/workspace/oclt', token='{hf_token}')"
echo "=== OCLT: Starting evaluation ==="
cd /workspace/oclt
PYTHONPATH=/workspace/oclt python -m worker.evaluate
"""


def _vllm_script(hf_model_repo: str, hf_token: str, vllm_cfg) -> str:
    return f"""#!/bin/bash
set -e
export HF_TOKEN="{hf_token}"
python -m vllm.entrypoints.openai.api_server --model {hf_model_repo} --host 0.0.0.0 --port 8000 --tensor-parallel-size {vllm_cfg.tensor_parallel_size} --max-model-len {vllm_cfg.max_model_len} --gpu-memory-utilization {vllm_cfg.gpu_memory_utilization} --trust-remote-code
"""


# The docker_args decode the base64 STARTUP_SCRIPT env var and execute it.
# This avoids all quoting/newline issues with RunPod's GraphQL API.
# RunPod's SDK wraps docker_args in double quotes in the GraphQL mutation:
#   dockerArgs: "..."
# This means docker_args CANNOT contain " or $ (GraphQL treats $ as variable).
# Solution: python one-liner using only single quotes — no " or $ anywhere.
_BOOTSTRAP_CMD = (
    "python3 -c '"
    "import os,base64,subprocess;"
    "s=base64.b64decode(os.environ.get(chr(83)+chr(84)+chr(65)+chr(82)+chr(84)+chr(85)+chr(80)+chr(95)+chr(83)+chr(67)+chr(82)+chr(73)+chr(80)+chr(84)));"
    "f=open(chr(47)+chr(116)+chr(109)+chr(112)+chr(47)+chr(115)+chr(46)+chr(115)+chr(104),chr(119)+chr(98));"
    "f.write(s);f.close();"
    "subprocess.run([chr(98)+chr(97)+chr(115)+chr(104),chr(47)+chr(116)+chr(109)+chr(112)+chr(47)+chr(115)+chr(46)+chr(115)+chr(104)])"
    "'"
)


def create_training_pod(
    settings: Settings,
    job_id: str,
    project_id: str,
    base_model: str,
    worker_repo: str,
    gpu_type: str | None = None,
    callback_url: str = "",
) -> dict:
    """Create a RunPod GPU pod for training using a standard base image."""
    _init_runpod(settings)

    gpu = gpu_type or settings.runpod.gpu_type_id
    script = _training_script(worker_repo, settings.hf_token)

    env_vars = {
        "STARTUP_SCRIPT": _encode_script(script),
        "JOB_TYPE": "train",
        "JOB_ID": job_id,
        "PROJECT_ID": project_id,
        "BASE_MODEL": base_model,
        "HF_TOKEN": settings.hf_token,
        "HF_USERNAME": settings.hf_username,
        "OPENAI_API_KEY": settings.openai_api_key,
        "REPO_PREFIX": settings.huggingface.repo_prefix,
        "QUANTIZATION": settings.training.quantization,
        "LORA_R": str(settings.training.lora.r),
        "LORA_ALPHA": str(settings.training.lora.lora_alpha),
        "LORA_DROPOUT": str(settings.training.lora.lora_dropout),
        "NUM_EPOCHS": str(settings.training.training_args.num_train_epochs),
        "BATCH_SIZE": str(settings.training.training_args.per_device_train_batch_size),
        "LEARNING_RATE": str(settings.training.training_args.learning_rate),
        "MAX_SEQ_LENGTH": str(settings.training.training_args.max_seq_length),
    }

    # Callback URL for real-time progress updates from worker to backend
    if callback_url:
        env_vars["CALLBACK_URL"] = f"{callback_url}/api/projects/{project_id}/train/callback"

    pod_config = {
        "name": f"oclt-train-{project_id}-{job_id[:6]}",
        "image_name": settings.runpod.training_image,
        "gpu_type_id": gpu,
        "cloud_type": settings.runpod.cloud_type,
        "volume_in_gb": 80,
        "container_disk_in_gb": 30,
        "env": env_vars,
        "docker_args": _BOOTSTRAP_CMD,
    }

    if settings.runpod_volume_id:
        pod_config["network_volume_id"] = settings.runpod_volume_id

    logger.info(f"Creating RunPod training pod: {pod_config['name']} on {gpu}")
    pod = runpod.create_pod(**pod_config)
    logger.info(f"Pod created: {pod}")
    return pod


def create_eval_pod(
    settings: Settings,
    job_id: str,
    project_id: str,
    base_model: str,
    hf_model_repo: str,
    worker_repo: str,
    gpu_type: str | None = None,
) -> dict:
    """Create a RunPod GPU pod for evaluation using a standard base image."""
    _init_runpod(settings)

    gpu = gpu_type or settings.runpod.gpu_type_id
    script = _eval_script(worker_repo, settings.hf_token)

    env_vars = {
        "STARTUP_SCRIPT": _encode_script(script),
        "JOB_TYPE": "evaluate",
        "JOB_ID": job_id,
        "PROJECT_ID": project_id,
        "BASE_MODEL": base_model,
        "FINETUNED_MODEL": hf_model_repo,
        "HF_TOKEN": settings.hf_token,
        "HF_USERNAME": settings.hf_username,
        "REPO_PREFIX": settings.huggingface.repo_prefix,
        "OPENAI_API_KEY": settings.openai_api_key,
    }

    pod_config = {
        "name": f"oclt-eval-{project_id}-{job_id[:6]}",
        "image_name": settings.runpod.training_image,
        "gpu_type_id": gpu,
        "cloud_type": settings.runpod.cloud_type,
        "volume_in_gb": 80,
        "container_disk_in_gb": 30,
        "env": env_vars,
        "docker_args": _BOOTSTRAP_CMD,
    }

    if settings.runpod_volume_id:
        pod_config["network_volume_id"] = settings.runpod_volume_id

    logger.info(f"Creating RunPod eval pod: {pod_config['name']}")
    pod = runpod.create_pod(**pod_config)
    return pod


def create_deployment_pod(
    settings: Settings,
    project_id: str,
    hf_model_repo: str,
    gpu_type: str | None = None,
) -> dict:
    """Create a RunPod GPU pod running vLLM for inference."""
    _init_runpod(settings)

    gpu = gpu_type or settings.runpod.gpu_type_id
    vllm_cfg = settings.deployment.vllm
    script = _vllm_script(hf_model_repo, settings.hf_token, vllm_cfg)

    env_vars = {
        "STARTUP_SCRIPT": _encode_script(script),
        "MODEL_NAME": hf_model_repo,
        "HF_TOKEN": settings.hf_token,
    }

    pod_config = {
        "name": f"oclt-serve-{project_id}",
        "image_name": settings.runpod.vllm_image,
        "gpu_type_id": gpu,
        "cloud_type": settings.runpod.cloud_type,
        "volume_in_gb": 100,
        "container_disk_in_gb": 30,
        "ports": "8000/http",
        "env": env_vars,
        "docker_args": _BOOTSTRAP_CMD,
    }

    if settings.runpod_volume_id:
        pod_config["network_volume_id"] = settings.runpod_volume_id

    logger.info(f"Creating RunPod vLLM deployment pod: {pod_config['name']}")
    pod = runpod.create_pod(**pod_config)
    return pod


def get_pod_status(settings: Settings, pod_id: str) -> dict:
    """Get the status of a RunPod pod."""
    _init_runpod(settings)
    pod = runpod.get_pod(pod_id)
    return pod


def terminate_pod(settings: Settings, pod_id: str):
    """Terminate a RunPod pod."""
    _init_runpod(settings)
    logger.info(f"Terminating RunPod pod: {pod_id}")
    runpod.terminate_pod(pod_id)


def stop_pod(settings: Settings, pod_id: str):
    """Stop (but not terminate) a RunPod pod."""
    _init_runpod(settings)
    logger.info(f"Stopping RunPod pod: {pod_id}")
    runpod.stop_pod(pod_id)
