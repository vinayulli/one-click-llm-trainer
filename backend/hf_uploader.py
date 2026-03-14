"""Upload models, datasets, and worker scripts to HuggingFace Hub."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from huggingface_hub import HfApi, create_repo

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"


def upload_model_to_hub(
    local_path: str,
    repo_name: str,
    hf_token: str,
    private: bool = True,
) -> str:
    """Upload a merged model directory to HuggingFace Hub."""
    api = HfApi(token=hf_token)

    logger.info(f"Creating HF repo: {repo_name}")
    create_repo(repo_name, token=hf_token, private=private, exist_ok=True)

    logger.info(f"Uploading model from {local_path} to {repo_name}")
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_name,
        commit_message="Upload fine-tuned model via One Click LLM Trainer",
    )

    repo_url = f"https://huggingface.co/{repo_name}"
    logger.info(f"Model uploaded: {repo_url}")
    return repo_url


def upload_dataset_to_hub(
    local_path: str,
    repo_name: str,
    hf_token: str,
    private: bool = True,
) -> str:
    """Upload a dataset directory to HuggingFace Hub."""
    api = HfApi(token=hf_token)

    create_repo(
        repo_name, token=hf_token, private=private,
        repo_type="dataset", exist_ok=True,
    )

    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_name,
        repo_type="dataset",
        commit_message="Upload training dataset via One Click LLM Trainer",
    )

    return f"https://huggingface.co/datasets/{repo_name}"


def upload_worker_scripts(
    repo_name: str,
    hf_token: str,
) -> str:
    """
    Upload the worker/ scripts to a HuggingFace repo so RunPod pods can pull them.
    This eliminates the need for users to build/push Docker images.
    """
    api = HfApi(token=hf_token)

    create_repo(repo_name, token=hf_token, private=True, exist_ok=True)

    # Upload each worker file individually into a worker/ folder in the repo
    worker_files = [
        "train.py",
        "evaluate.py",
        "utils.py",
        "__init__.py",
        "requirements.txt",
    ]

    for fname in worker_files:
        fpath = WORKER_DIR / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"worker/{fname}",
                repo_id=repo_name,
                commit_message=f"Upload worker/{fname}",
            )
            logger.info(f"Uploaded worker/{fname} to {repo_name}")

    repo_url = f"https://huggingface.co/{repo_name}"
    logger.info(f"Worker scripts uploaded: {repo_url}")
    return repo_url
