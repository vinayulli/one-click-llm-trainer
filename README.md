# One Click LLM Trainer

Fine-tune LLMs on domain-specific data with minimal human intervention. Upload documents, auto-generate instruction datasets, auto-select the best model, train on RunPod GPUs, evaluate, and deploy — all through a single API.

**Zero Docker setup required** — the app automatically provisions RunPod pods using standard base images and handles all code deployment via HuggingFace.

## Architecture

```
[User / Frontend]
        |
  [FastAPI Backend]     <- orchestrator (no GPU needed)
        |
  +-----+------+--------+-----------+
  |            |         |           |
[RunPod     [RunPod    [RunPod     [HuggingFace
 Training    Eval       vLLM        Hub]
 Pod]        Pod]       Deployment]
```

**How it works without Docker:**
1. Worker scripts (`worker/train.py`, `worker/evaluate.py`) are uploaded to a private HuggingFace repo
2. RunPod pods use standard base images (`runpod/pytorch`, `vllm/vllm-openai`)
3. A startup script on each pod installs dependencies, pulls code from HF, and runs the job
4. No Docker build, no container registry, no DevOps — fully automatic

## Quick Start

### 1. Install & Configure

```bash
git clone <repo-url>
cd one-click-llm-trainer
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

Required keys:
| Key | Where to get it |
|-----|----------------|
| `OCLT_RUNPOD_API_KEY` | https://runpod.io/console/user/settings |
| `OCLT_OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `OCLT_HF_TOKEN` | https://huggingface.co/settings/tokens |
| `OCLT_HF_USERNAME` | Your HuggingFace username |

### 2. Run

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 for the web UI, or http://localhost:8000/docs for the API.

## Pipeline Flow

The web UI at `http://localhost:8000` guides you through each step:

| Step | Action | What happens |
|------|--------|-------------|
| 1 | Create Project | Name your fine-tuning project |
| 2 | Upload Docs | Drag & drop PDF/DOCX/TXT files, auto-chunked |
| 3 | Generate Dataset | GPT-4o-mini converts chunks into instruction/input/output examples |
| 4 | Select Model | System auto-ranks models, you confirm or pick another |
| 5 | Train | RunPod GPU pod spins up, runs QLoRA, uploads model to HF |
| 6 | Evaluate | Base vs fine-tuned comparison (exact match, F1, LLM judge) |
| 7 | Deploy | One-click vLLM deployment on RunPod + sample code |

## API Endpoints

```
POST   /api/projects                          Create project
POST   /api/projects/{id}/upload              Upload domain documents
POST   /api/projects/{id}/process             Chunk documents
POST   /api/projects/{id}/generate-dataset    Generate instruction dataset
GET    /api/projects/{id}/suggest-models      Auto model suggestions
POST   /api/projects/{id}/train               Start training on RunPod
GET    /api/projects/{id}/train/status        Poll training progress
POST   /api/projects/{id}/train/cancel        Cancel training
POST   /api/projects/{id}/evaluate            Start evaluation on RunPod
GET    /api/projects/{id}/evaluate/results    Get comparison results
POST   /api/projects/{id}/deploy              Deploy with vLLM on RunPod
GET    /api/projects/{id}/deploy/status       Deployment status + endpoint URL
GET    /api/projects/{id}/deploy/sample-code  Ready-to-use code snippets
DELETE /api/projects/{id}/deploy              Stop deployment
```

## Supported Base Models (27 models, 7 families)

| Family | Model | Size | Best For |
|--------|-------|------|----------|
| **Qwen 3** | Qwen 3 0.6B | 0.6B | Edge deployment, tiny datasets |
| | Qwen 3 1.7B | 1.7B | Budget training, multilingual |
| | Qwen 3 4B | 4B | Best quality/cost ratio, math (MATH-500: 97.0) |
| | Qwen 3 8B | 8B | Best overall sub-10B, 119 languages |
| **Qwen 2.5** | Qwen 2.5 0.5B Instruct | 0.5B | Prototyping, tiny datasets |
| | Qwen 2.5 1.5B Instruct | 1.5B | Small model, multilingual |
| | Qwen 2.5 3B Instruct | 3B | General purpose, affordable |
| | Qwen 2.5 7B Instruct | 7B | Proven workhorse, 128K context |
| | Qwen 2.5 Coder 1.5B | 1.5B | Code, small datasets |
| | Qwen 2.5 Coder 3B | 3B | Code generation |
| | Qwen 2.5 Coder 7B | 7B | Top-tier code fine-tuning |
| **Gemma 3** | Gemma 3 1B IT | 1B | Lightweight multilingual (140+ langs) |
| | Gemma 3 4B IT | 4B | Multilingual, code (HumanEval 71.3) |
| **Gemma 2** | Gemma 2 2B IT | 2.6B | Simple English tasks |
| | Gemma 2 9B IT | 9B | Instruction following |
| **Llama 3.2** | Llama 3.2 1B | 1B | On-device, 128K context |
| | Llama 3.2 1B Instruct | 1B | Lightweight instruction following |
| | Llama 3.2 3B | 3B | Tool use, agentic workflows |
| | Llama 3.2 3B Instruct | 3B | Multilingual dialogue |
| **Llama 3.1** | Llama 3.1 8B | 8B | General purpose, code (HumanEval 72.6) |
| | Llama 3.1 8B Instruct | 8B | Tool use, instruction following |
| **Phi** | Phi 3.5 Mini | 3.8B | Summarization, reasoning |
| | Phi 4 Mini | 3.8B | Best coding at this size (HumanEval 74.4) |
| **SmolLM2** | SmolLM2 1.7B Instruct | 1.7B | Compact general purpose |
| | SmolLM2 360M Instruct | 360M | Ultra-tiny, edge/classification |

Auto-selection criteria: dataset size, language detection, domain detection, GPU cost.

## Example Outputs

- **Example Instruction Dataset**: https://huggingface.co/datasets/vinaybabu/oclt-data-2aba2de4dcc6
- **Example Fine-tuned Model**: https://huggingface.co/vinaybabu/oclt-87c32cac2338

## Project Structure

```
one-click-llm-trainer/
├── backend/                  # FastAPI orchestrator (no GPU needed)
│   ├── main.py              # App entry + frontend serving
│   ├── config.py            # Settings + YAML config loading
│   ├── models.py            # Pydantic schemas
│   ├── storage.py           # SQLite job persistence
│   ├── document_processor.py # PDF/DOCX/TXT extraction + chunking
│   ├── dataset_generator.py  # LLM-powered dataset creation
│   ├── model_selector.py     # Auto model suggestion engine
│   ├── trainer.py            # Training orchestrator (RunPod)
│   ├── evaluator.py          # Eval orchestrator (RunPod)
│   ├── deployer.py           # Deploy orchestrator (RunPod + vLLM)
│   ├── runpod_client.py      # RunPod SDK + startup script generation
│   ├── hf_uploader.py        # HF Hub upload (models, data, worker scripts)
│   ├── scheduler.py          # Background job status poller
│   └── routers/              # API route handlers
├── worker/                   # Uploaded to HF, pulled by RunPod pods at runtime
│   ├── train.py             # QLoRA training + merge + HF upload
│   ├── evaluate.py          # Base vs fine-tuned comparison
│   └── utils.py             # Shared utilities
├── frontend/                 # Web UI
│   ├── index.html           # Step-by-step wizard
│   └── static/              # CSS + JS
├── configs/
│   ├── default.yaml         # Default configuration
│   └── model_catalog.yaml   # Supported models + GPU pricing
├── Dockerfile               # Backend container (optional)
├── Dockerfile.vllm          # vLLM reference (used by RunPod directly)
└── requirements.txt
```

## How RunPod Provisioning Works (Zero Docker)

When you click "Start Training", the app:

1. **Uploads worker scripts** to `{username}/oclt-worker` on HuggingFace (private repo)
2. **Uploads dataset** to `{username}/oclt-data-{project_id}` on HuggingFace
3. **Creates a RunPod pod** using `runpod/pytorch:2.2.0` (standard image, no build needed)
4. **Pod startup script** automatically:
   - `pip install` all ML dependencies (transformers, peft, trl, bitsandbytes, etc.)
   - Downloads worker scripts from HuggingFace
   - Runs `worker/train.py`
5. **Training completes** → merged model uploaded to `{username}/oclt-{project_id}` on HF
6. **Pod self-terminates** → no ongoing GPU cost

Same flow for evaluation and deployment (vLLM uses its own standard image).
