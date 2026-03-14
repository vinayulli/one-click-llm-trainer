from __future__ import annotations

import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class RunPodConfig(BaseModel):
    api_key: str = ""
    gpu_type_id: str = "NVIDIA RTX A5000"
    volume_id: str = ""
    cloud_type: str = "COMMUNITY"  # COMMUNITY or SECURE
    max_bid_per_gpu: float = 0.5
    # Standard RunPod base images — no custom Docker build required
    training_image: str = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
    vllm_image: str = "vllm/vllm-openai:latest"


class HuggingFaceConfig(BaseModel):
    token: str = ""
    username: str = ""
    repo_prefix: str = "oclt"
    private: bool = True


class DocumentProcessingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 100
    supported_formats: list[str] = [".pdf", ".docx", ".txt"]


class DatasetGenConfig(BaseModel):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    max_examples_per_chunk: int = 3
    train_split: float = 0.8
    val_split: float = 0.1
    eval_split: float = 0.1


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]


class TrainingArgsConfig(BaseModel):
    num_train_epochs: float = 0.5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    fp16: bool = True


class TrainingConfig(BaseModel):
    quantization: str = "4bit"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training_args: TrainingArgsConfig = Field(default_factory=TrainingArgsConfig)


class VLLMConfig(BaseModel):
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9


class DeploymentConfig(BaseModel):
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    port: int = 8001


# ---------------------------------------------------------------------------
# Main settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    project_name: str = "my_project"
    base_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    db_url: str = "sqlite+aiosqlite:///./oclt.db"

    # API keys
    openai_api_key: str = ""
    hf_token: str = ""
    hf_username: str = ""
    runpod_api_key: str = ""
    runpod_volume_id: str = ""

    # Public URL of this backend (for RunPod worker callbacks)
    # e.g. https://your-server.com or ngrok URL
    public_url: str = ""

    # Sub-configs
    runpod: RunPodConfig = Field(default_factory=RunPodConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    document_processing: DocumentProcessingConfig = Field(
        default_factory=DocumentProcessingConfig
    )
    dataset_generation: DatasetGenConfig = Field(default_factory=DatasetGenConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    model_config = {"env_prefix": "OCLT_", "env_file": ".env", "extra": "ignore"}

    def project_raw_dir(self, project_id: str) -> Path:
        return self.base_dir / "raw" / project_id

    def project_processed_dir(self, project_id: str) -> Path:
        return self.base_dir / "processed" / project_id

    def project_dataset_dir(self, project_id: str) -> Path:
        return self.base_dir / "datasets" / project_id

    def project_model_dir(self, project_id: str) -> Path:
        return self.models_dir / project_id


def load_settings(config_path: str | None = None) -> Settings:
    """Load settings, optionally merging a YAML config file."""
    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        project = data.get("project", {})
        kwargs: dict = {}

        if "name" in project:
            kwargs["project_name"] = project["name"]
        if "base_dir" in project:
            kwargs["base_dir"] = project["base_dir"]

        for section, cls in [
            ("runpod", RunPodConfig),
            ("huggingface", HuggingFaceConfig),
            ("document_processing", DocumentProcessingConfig),
            ("dataset_generation", DatasetGenConfig),
            ("training", TrainingConfig),
            ("deployment", DeploymentConfig),
        ]:
            if section in data:
                kwargs[section] = cls(**data[section])

        return Settings(**kwargs)
    return Settings()


settings = load_settings()
