"""Shared utilities for RunPod worker scripts."""

from __future__ import annotations

import json
import os
import time
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
_CALLBACK_URL = os.environ.get("CALLBACK_URL", "")
_start_time: float | None = None


def write_status(phase: str, **kwargs):
    """
    Write status to local file AND post to backend callback URL.
    The callback URL allows the backend to track real-time training progress
    without needing to read files from the RunPod pod.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    # Calculate ETA if we have step progress
    step = kwargs.get("step", 0)
    total_steps = kwargs.get("total_steps", 0)
    if step > 0 and total_steps > 0:
        elapsed = time.time() - _start_time
        steps_per_sec = step / max(elapsed, 1)
        remaining_steps = total_steps - step
        eta_seconds = int(remaining_steps / max(steps_per_sec, 0.001))
        kwargs["eta_seconds"] = eta_seconds
        kwargs["eta_formatted"] = _format_eta(eta_seconds)
        kwargs["elapsed_seconds"] = int(elapsed)
        kwargs["elapsed_formatted"] = _format_eta(int(elapsed))
        kwargs["steps_per_second"] = round(steps_per_sec, 2)

    status = {"phase": phase, **kwargs}

    # Write to local file
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f)
    except Exception as e:
        logger.warning(f"Could not write status file: {e}")

    # POST to backend callback
    if _CALLBACK_URL:
        try:
            import urllib.request
            data = json.dumps(status).encode("utf-8")
            req = urllib.request.Request(
                _CALLBACK_URL,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.warning(f"Callback POST failed: {e}")

    logger.info(f"Status: {phase} | {kwargs}")


def _format_eta(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    hours = seconds // 3600
    mins = (seconds % 3600) // 60
    return f"{hours}h {mins}m"
