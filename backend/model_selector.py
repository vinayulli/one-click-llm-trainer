"""Auto-suggest and auto-select the best base model for fine-tuning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from backend.models import ModelSuggestion


CATALOG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model_catalog.yaml"


def _load_catalog() -> list[dict]:
    with open(CATALOG_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def _detect_language(texts: list[str]) -> str:
    """Heuristic language detection from text samples."""
    combined = " ".join(texts[:50]).lower()

    # CJK character ranges
    cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", combined))
    latin_count = len(re.findall(r"[a-z]", combined))

    if cjk_count > latin_count * 0.3:
        return "multilingual"
    return "en"


def _detect_domain(texts: list[str]) -> list[str]:
    """Heuristic domain detection from text samples."""
    combined = " ".join(texts[:100]).lower()
    domains = ["general"]

    code_indicators = [
        "def ", "class ", "import ", "function ", "const ", "var ",
        "return ", "if (", "for (", "```", "=>", "->",
    ]
    if sum(1 for ind in code_indicators if ind in combined) >= 3:
        domains.append("code")

    math_indicators = [
        "equation", "theorem", "proof", "integral", "derivative",
        "matrix", "∑", "∫", "≤", "≥", "probability",
    ]
    if sum(1 for ind in math_indicators if ind in combined) >= 2:
        domains.append("math")

    return domains


def suggest_models(
    num_examples: int,
    sample_texts: list[str],
    avg_instruction_length: float = 0.0,
    avg_output_length: float = 0.0,
) -> list[ModelSuggestion]:
    """
    Score and rank base models based on dataset characteristics.
    Returns top-3 suggestions with the first being the auto-selected best.
    """
    catalog = _load_catalog()
    language = _detect_language(sample_texts)
    domains = _detect_domain(sample_texts)

    logger.info(f"Dataset: {num_examples} examples, language={language}, domains={domains}")

    scored: list[tuple[dict, float, list[str]]] = []

    for model in catalog:
        score = 0.0
        reasons: list[str] = []
        params = model["param_billions"]

        # --- Dataset size fit ---
        if num_examples < 200:
            # Very small dataset: tiny models to avoid overfitting
            if params <= 2:
                score += 35
                reasons.append("Tiny model prevents overfitting on very small dataset (<200 examples)")
            elif params <= 4:
                score += 25
                reasons.append("Small model suitable for small dataset")
            elif params <= 8:
                score += 10
            else:
                score -= 5
        elif num_examples < 500:
            if params <= 4:
                score += 30
                reasons.append("Small model suits small dataset (<500 examples)")
            elif params <= 8:
                score += 20
                reasons.append("Medium model can work with small dataset")
        elif num_examples < 1500:
            if 3 <= params <= 8:
                score += 25
                reasons.append("Good model size for medium dataset")
            elif params < 3:
                score += 10
        else:
            # Large dataset: bigger models can leverage more data
            score += 15 + params * 2
            reasons.append("Larger dataset supports bigger models")

        # --- Language fit ---
        if language == "multilingual":
            if "multilingual" in model.get("domains", []):
                score += 30
                reasons.append("Multilingual model matches non-English content")
            else:
                score -= 15
        else:
            score += 5  # English is fine for all models

        # --- Domain fit ---
        for domain in domains:
            if domain != "general" and domain in model.get("domains", []):
                score += 20
                reasons.append(f"Specialized for '{domain}' domain")
            elif domain == "general" and "general" in model.get("domains", []):
                score += 5

        # --- Instruction-tuned preference ---
        mid = model["model_id"].lower()
        if "instruct" in mid or mid.endswith("-it"):
            score += 12
            reasons.append("Instruction-tuned base improves fine-tuning results")

        # --- Generation preference (newer > older) ---
        if "qwen3" in mid:
            score += 15
            reasons.append("Latest generation Qwen3 (119 languages, dual thinking)")
        elif "gemma-3" in mid:
            score += 12
            reasons.append("Latest Gemma3 (140+ languages)")
        elif "phi-4" in mid:
            score += 12
            reasons.append("Latest Phi-4 (best coding at this size)")
        elif "llama-3.2" in mid:
            score += 8
            reasons.append("Recent Llama 3.2 release")
        elif "qwen2.5" in mid:
            score += 6

        # --- Practical: cost efficiency ---
        if model["min_gpu_vram_gb"] <= 16:
            score += 12
            reasons.append("Runs on budget GPU (16GB, ~$0.16/hr)")
        elif model["min_gpu_vram_gb"] <= 24:
            score += 8
            reasons.append("Fits on mid-tier GPU (24GB)")

        # --- Context length bonus ---
        if avg_output_length > 300 and model["context_length"] >= 32768:
            score += 5
            reasons.append("Long context supports detailed outputs")
        elif model["context_length"] < 8192:
            score -= 5
            reasons.append("Short context (8K) may limit output quality")

        scored.append((model, score, reasons))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return top 5 to give users more choices with the expanded catalog
    suggestions = []
    for model, score, reasons in scored[:5]:
        # Estimate training time (rough: ~1 min per 100 examples per billion params)
        est_hours = (num_examples / 100) * (model["param_billions"] / 7.0) * (1 / 60)
        est_hours = max(0.1, round(est_hours, 2))
        est_cost = round(est_hours * model["cost_per_hour"], 2)

        suggestions.append(
            ModelSuggestion(
                model_id=model["model_id"],
                display_name=model["display_name"],
                parameter_count=model["parameter_count"],
                score=round(score, 1),
                reasoning="; ".join(reasons),
                recommended_gpu=model["recommended_gpu"],
                estimated_train_time_hours=est_hours,
                estimated_cost_usd=est_cost,
            )
        )

    return suggestions
