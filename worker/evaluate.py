"""
Evaluation worker — runs inside a RunPod GPU container.

Compares base model vs fine-tuned model on the eval split.
Metrics: exact match, F1 score, LLM judge score.
"""

from __future__ import annotations

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login
from loguru import logger
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from worker.utils import get_env, write_status


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def exact_match_score(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def llm_judge_score(
    instruction: str,
    reference: str,
    prediction: str,
    client: OpenAI,
) -> float:
    prompt = f"""Rate the quality of the following AI response on a scale of 1-10.

Instruction: {instruction}
Reference Answer: {reference}
Model Response: {prediction}

Consider accuracy, completeness, and relevance.
Return ONLY a JSON object with "score" (integer 1-10) and "reasoning" (brief explanation)."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return float(result.get("score", 5)) / 10.0


def generate_response(
    model, tokenizer, instruction: str, input_text: str = "",
    max_new_tokens: int = 512,
) -> str:
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return generated.strip()


def main():
    project_id = get_env("PROJECT_ID")
    base_model_name = get_env("BASE_MODEL")
    finetuned_model = get_env("FINETUNED_MODEL")  # HF repo
    hf_token = get_env("HF_TOKEN")
    openai_api_key = get_env("OPENAI_API_KEY")
    hf_username = get_env("HF_USERNAME", "")
    repo_prefix = get_env("REPO_PREFIX", "oclt")

    dataset_repo = f"{hf_username}/{repo_prefix}-data-{project_id}" if hf_username else finetuned_model.replace(repo_prefix + "-", repo_prefix + "-data-")

    if hf_token:
        login(token=hf_token)

    output_dir = Path("/workspace/eval_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    write_status(phase="loading_eval_data")

    # Load eval dataset
    logger.info(f"Loading eval dataset from: {dataset_repo}")
    eval_dataset = load_dataset(dataset_repo, data_files="eval.jsonl", split="train")
    logger.info(f"Eval examples: {len(eval_dataset)}")

    # Load base model
    write_status(phase="loading_base_model")
    logger.info(f"Loading base model: {base_model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    # Load fine-tuned model
    write_status(phase="loading_finetuned_model")
    logger.info(f"Loading fine-tuned model: {finetuned_model}")

    ft_tokenizer = AutoTokenizer.from_pretrained(
        finetuned_model, trust_remote_code=True, token=hf_token,
    )
    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token

    # Clear GPU memory and load fine-tuned model
    ft_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )

    openai_client = None
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)

    # --- Run evaluation ---
    write_status(phase="evaluating", total=len(eval_dataset))
    results = []
    base_metrics = {"exact_match": [], "f1": [], "llm_judge": []}
    ft_metrics = {"exact_match": [], "f1": [], "llm_judge": []}

    for i, example in enumerate(eval_dataset):
        logger.info(f"Evaluating {i + 1}/{len(eval_dataset)}")
        write_status(
            phase="evaluating",
            current=i + 1,
            total=len(eval_dataset),
            progress_pct=round((i + 1) / len(eval_dataset) * 100, 1),
        )

        instruction = example["instruction"]
        input_text = example.get("input", "")
        reference = example["output"]

        # Base model response
        base_response = generate_response(base_model, tokenizer, instruction, input_text)

        # Fine-tuned model response
        ft_response = generate_response(ft_model, ft_tokenizer, instruction, input_text)

        entry = {
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "base_response": base_response,
            "finetuned_response": ft_response,
            "base_metrics": {},
            "finetuned_metrics": {},
        }

        for label, response, metrics_dict in [
            ("base_metrics", base_response, base_metrics),
            ("finetuned_metrics", ft_response, ft_metrics),
        ]:
            em = exact_match_score(response, reference)
            f1 = f1_score(response, reference)
            entry[label]["exact_match"] = em
            entry[label]["f1"] = f1
            metrics_dict["exact_match"].append(em)
            metrics_dict["f1"].append(f1)

            if openai_client:
                try:
                    judge = llm_judge_score(instruction, reference, response, openai_client)
                    entry[label]["llm_judge"] = judge
                    metrics_dict["llm_judge"].append(judge)
                except Exception as e:
                    logger.warning(f"LLM judge failed: {e}")

        results.append(entry)

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary = {
        "num_examples": len(eval_dataset),
        "base_model": base_model_name,
        "finetuned_model": finetuned_model,
        "base_avg": {k: avg(v) for k, v in base_metrics.items() if v},
        "finetuned_avg": {k: avg(v) for k, v in ft_metrics.items() if v},
        "details": results,
    }

    report_path = output_dir / "eval_results.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {report_path}")
    logger.info(f"Base model avg:       {summary['base_avg']}")
    logger.info(f"Fine-tuned model avg: {summary['finetuned_avg']}")

    write_status(
        phase="completed",
        base_avg=summary["base_avg"],
        finetuned_avg=summary["finetuned_avg"],
        num_examples=len(eval_dataset),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        write_status(phase="failed", error=str(e))
        sys.exit(1)
