"""Shared utilities for RunPod worker scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger


def format_instruction(example: dict) -> str:
    """Format an instruction-tuning example into a prompt string."""
    if example.get("input"):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )


def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def get_env_int(key: str, default: int = 0) -> int:
    return int(os.environ.get(key, str(default)))


def get_env_float(key: str, default: float = 0.0) -> float:
    return float(os.environ.get(key, str(default)))


STATUS_FILE = Path("/workspace/status.json")


def write_status(phase: str, **kwargs):
    """Write status to a file on the network volume for polling."""
    status = {"phase": phase, **kwargs}
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f)
        logger.info(f"Status: {phase} | {kwargs}")
    except Exception as e:
        logger.warning(f"Could not write status file: {e}")
