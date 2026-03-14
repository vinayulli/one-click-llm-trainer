"""Generate instruction-tuning dataset from document chunks using an LLM."""

from __future__ import annotations

import json
import random
from pathlib import Path

from loguru import logger
from openai import OpenAI

from backend.config import Settings

SYSTEM_PROMPT = """You are an expert at creating instruction-tuning datasets for LLMs.
Given a text chunk from a domain document, generate high-quality instruction-input-output examples.

Rules:
- Each example must be grounded in the provided text.
- Instructions should be diverse: questions, summaries, explanations, comparisons, etc.
- The output must be accurate and based only on the given text.
- Return valid JSON array of objects with keys: instruction, input, output.
- Do NOT include any text outside the JSON array."""

USER_TEMPLATE = """Generate exactly {n} instruction-tuning examples from this text chunk:

---
{text}
---

Return a JSON array of {n} objects, each with "instruction", "input", and "output" keys."""


def generate_examples_from_chunk(
    text: str,
    client: OpenAI,
    model: str,
    max_examples: int = 3,
) -> list[dict]:
    prompt = USER_TEMPLATE.format(n=max_examples, text=text)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)

    # Handle both {"examples": [...]} and [...] formats
    if isinstance(parsed, dict):
        for key in ("examples", "data", "items"):
            if key in parsed:
                parsed = parsed[key]
                break
        else:
            parsed = list(parsed.values())[0] if parsed else []

    if not isinstance(parsed, list):
        parsed = [parsed]

    validated = []
    for item in parsed:
        if all(k in item for k in ("instruction", "input", "output")):
            validated.append({
                "instruction": str(item["instruction"]),
                "input": str(item["input"]),
                "output": str(item["output"]),
            })
    return validated


def compute_dataset_stats(examples: list[dict]) -> dict:
    """Compute statistics for model selection."""
    if not examples:
        return {"total": 0}

    inst_lengths = [len(e["instruction"]) for e in examples]
    out_lengths = [len(e["output"]) for e in examples]
    total_chars = sum(inst_lengths) + sum(out_lengths)

    return {
        "total_examples": len(examples),
        "avg_instruction_length": sum(inst_lengths) / len(inst_lengths),
        "avg_output_length": sum(out_lengths) / len(out_lengths),
        "estimated_tokens": total_chars // 4,  # rough estimate
    }


def generate_dataset(project_id: str, settings: Settings) -> dict:
    cfg = settings.dataset_generation
    processed_dir = settings.project_processed_dir(project_id)
    dataset_dir = settings.project_dataset_dir(project_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = processed_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError("No processed chunks found. Upload and process documents first.")

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    client = OpenAI(api_key=settings.openai_api_key)
    all_examples: list[dict] = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Generating examples for chunk {i + 1}/{len(chunks)}")
        try:
            examples = generate_examples_from_chunk(
                text=chunk["text"],
                client=client,
                model=cfg.llm_model,
                max_examples=cfg.max_examples_per_chunk,
            )
            for ex in examples:
                ex["source"] = chunk.get("source", "unknown")
                ex["chunk_id"] = chunk.get("chunk_id", i)
            all_examples.extend(examples)
        except Exception as e:
            logger.error(f"Failed on chunk {i}: {e}")
            continue

    # Compute stats before shuffling
    stats = compute_dataset_stats(all_examples)

    random.shuffle(all_examples)
    total = len(all_examples)
    train_end = int(total * cfg.train_split)
    val_end = train_end + int(total * cfg.val_split)

    splits = {
        "train": all_examples[:train_end],
        "validation": all_examples[train_end:val_end],
        "eval": all_examples[val_end:],
    }

    paths = {}
    for split_name, split_data in splits.items():
        path = dataset_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        paths[split_name] = str(path)
        logger.info(f"{split_name}: {len(split_data)} examples -> {path}")

    # Save stats
    stats.update({
        "train_count": len(splits["train"]),
        "validation_count": len(splits["validation"]),
        "eval_count": len(splits["eval"]),
    })
    with open(dataset_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return {
        "total_examples": total,
        "stats": stats,
        "splits": {k: len(v) for k, v in splits.items()},
        "paths": paths,
        "sample_texts": [e.get("output", "") for e in all_examples[:50]],
    }
