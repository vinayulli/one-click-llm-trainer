import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.dataset_generator import generate_dataset
from backend.model_selector import suggest_models
from backend.models import DatasetResponse, DatasetStats, ModelSuggestionResponse
from backend.storage import get_project, update_project_config, update_project_stage

router = APIRouter(prefix="/api/projects/{project_id}", tags=["datasets"])


@router.post("/generate-dataset", response_model=DatasetResponse)
async def generate_instruction_dataset(project_id: str):
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = generate_dataset(project_id, settings)

    stats = DatasetStats(
        total_examples=result["total_examples"],
        train_count=result["splits"]["train"],
        validation_count=result["splits"]["validation"],
        eval_count=result["splits"]["eval"],
        avg_instruction_length=result["stats"].get("avg_instruction_length", 0),
        avg_output_length=result["stats"].get("avg_output_length", 0),
        estimated_tokens=result["stats"].get("estimated_tokens", 0),
    )

    await update_project_stage(project_id, "dataset_generated")
    await update_project_config(project_id, {"dataset_stats": result["stats"]})

    # Load first few examples for preview
    preview = []
    train_path = Path(result["paths"]["train"])
    if train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                preview.append(json.loads(line))

    return DatasetResponse(
        project_id=project_id,
        stats=stats,
        preview=preview,
    )


@router.get("/dataset", response_model=DatasetResponse)
async def get_dataset_info(project_id: str):
    dataset_dir = settings.project_dataset_dir(project_id)
    stats_path = dataset_dir / "stats.json"

    if not stats_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not generated yet")

    with open(stats_path) as f:
        raw_stats = json.load(f)

    stats = DatasetStats(
        total_examples=raw_stats.get("total_examples", 0),
        train_count=raw_stats.get("train_count", 0),
        validation_count=raw_stats.get("validation_count", 0),
        eval_count=raw_stats.get("eval_count", 0),
        avg_instruction_length=raw_stats.get("avg_instruction_length", 0),
        avg_output_length=raw_stats.get("avg_output_length", 0),
        estimated_tokens=raw_stats.get("estimated_tokens", 0),
    )

    preview = []
    train_path = dataset_dir / "train.jsonl"
    if train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                preview.append(json.loads(line))

    return DatasetResponse(project_id=project_id, stats=stats, preview=preview)


@router.get("/suggest-models", response_model=ModelSuggestionResponse)
async def get_model_suggestions(project_id: str):
    dataset_dir = settings.project_dataset_dir(project_id)
    stats_path = dataset_dir / "stats.json"

    if not stats_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not generated yet")

    with open(stats_path) as f:
        raw_stats = json.load(f)

    # Collect sample texts from the training data
    sample_texts = []
    train_path = dataset_dir / "train.jsonl"
    if train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                ex = json.loads(line)
                sample_texts.append(ex.get("output", ""))

    train_cfg = settings.training.training_args
    suggestions = suggest_models(
        num_examples=raw_stats.get("total_examples", 0),
        sample_texts=sample_texts,
        avg_instruction_length=raw_stats.get("avg_instruction_length", 0),
        avg_output_length=raw_stats.get("avg_output_length", 0),
        epochs=train_cfg.num_train_epochs,
        batch_size=train_cfg.per_device_train_batch_size,
        grad_accum=train_cfg.gradient_accumulation_steps,
    )

    await update_project_stage(project_id, "model_suggested")

    return ModelSuggestionResponse(
        project_id=project_id,
        suggestions=suggestions,
        auto_selected=suggestions[0].model_id if suggestions else "",
    )
