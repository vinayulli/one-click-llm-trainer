"""
Training worker — runs inside a RunPod GPU container.

Workflow:
1. Download dataset from HuggingFace
2. Load base model with QLoRA
3. Fine-tune with SFTTrainer
4. Merge LoRA adapter with base model
5. Upload merged model to HuggingFace Hub
6. Write completion status
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, login
from loguru import logger
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from worker.utils import (
    format_instruction,
    get_env,
    get_env_float,
    get_env_int,
    write_status,
)


class StatusCallback(TrainerCallback):
    """Report training progress to the status file."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step > 0:
            write_status(
                phase="training",
                step=state.global_step,
                total_steps=state.max_steps,
                epoch=round(state.epoch, 2) if state.epoch else 0,
                loss=round(logs.get("loss", 0), 4) if logs else 0,
                progress_pct=round(state.global_step / max(state.max_steps, 1) * 100, 1),
            )


def main():
    # --- Read config from environment ---
    project_id = get_env("PROJECT_ID")
    base_model = get_env("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    hf_token = get_env("HF_TOKEN")
    hf_username = get_env("HF_USERNAME")
    quantization = get_env("QUANTIZATION", "4bit")
    lora_r = get_env_int("LORA_R", 16)
    lora_alpha = get_env_int("LORA_ALPHA", 32)
    lora_dropout = get_env_float("LORA_DROPOUT", 0.05)
    num_epochs = get_env_int("NUM_EPOCHS", 3)
    batch_size = get_env_int("BATCH_SIZE", 4)
    learning_rate = get_env_float("LEARNING_RATE", 2e-4)
    max_seq_length = get_env_int("MAX_SEQ_LENGTH", 2048)

    repo_prefix = get_env("REPO_PREFIX", "oclt")
    dataset_repo = f"{hf_username}/{repo_prefix}-data-{project_id}"
    model_repo = f"{hf_username}/{repo_prefix}-{project_id}"

    output_dir = Path("/workspace/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"

    # --- Login to HuggingFace ---
    if hf_token:
        login(token=hf_token)

    write_status(phase="downloading_dataset", dataset_repo=dataset_repo)

    # --- Load dataset ---
    logger.info(f"Loading dataset from: {dataset_repo}")
    train_dataset = load_dataset(dataset_repo, data_files="train.jsonl", split="train")
    val_dataset = None
    try:
        val_dataset = load_dataset(dataset_repo, data_files="validation.jsonl", split="train")
    except Exception:
        logger.warning("No validation split found, continuing without it")

    # Format dataset
    train_dataset = train_dataset.map(
        lambda x: {"text": format_instruction(x)},
        remove_columns=train_dataset.column_names,
    )
    if val_dataset:
        val_dataset = val_dataset.map(
            lambda x: {"text": format_instruction(x)},
            remove_columns=val_dataset.column_names,
        )

    logger.info(f"Training examples: {len(train_dataset)}")

    # --- Load model with quantization ---
    write_status(phase="loading_model", model=base_model)
    logger.info(f"Loading base model: {base_model}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(quantization == "4bit"),
        load_in_8bit=(quantization == "8bit"),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)

    # --- Configure LoRA ---
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # --- Training ---
    write_status(phase="training", step=0, total_steps=0, epoch=0)

    sft_config = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        max_seq_length=max_seq_length,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        callbacks=[StatusCallback()],
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Loss: {train_result.training_loss:.4f}")

    # Save adapter
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_status(phase="adapter_saved", loss=round(train_result.training_loss, 4))

    # --- Merge adapter with base model ---
    write_status(phase="merging_model")
    logger.info("Merging LoRA adapter with base model...")

    # Reload in float16 for merging (need more memory but cleaner merge)
    del model
    del trainer
    torch.cuda.empty_cache()

    from peft import PeftModel

    base_model_for_merge = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    merged_model = PeftModel.from_pretrained(base_model_for_merge, str(adapter_dir))
    merged_model = merged_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    logger.info(f"Merged model saved to {merged_dir}")

    # --- Upload to HuggingFace ---
    write_status(phase="uploading_to_hf", repo=model_repo)
    logger.info(f"Uploading merged model to HuggingFace: {model_repo}")

    api = HfApi(token=hf_token)
    create_repo(model_repo, token=hf_token, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=model_repo,
        commit_message=f"Fine-tuned {base_model} on project {project_id}",
    )

    hf_url = f"https://huggingface.co/{model_repo}"
    logger.info(f"Model uploaded: {hf_url}")

    # --- Save training metadata ---
    metadata = {
        "base_model": base_model,
        "model_repo": model_repo,
        "hf_url": hf_url,
        "dataset_repo": dataset_repo,
        "train_loss": round(train_result.training_loss, 4),
        "train_samples": len(train_dataset),
        "epochs": num_epochs,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "quantization": quantization,
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    write_status(
        phase="completed",
        hf_url=hf_url,
        model_repo=model_repo,
        train_loss=round(train_result.training_loss, 4),
    )
    logger.info("Training worker finished successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        write_status(phase="failed", error=str(e))
        sys.exit(1)
