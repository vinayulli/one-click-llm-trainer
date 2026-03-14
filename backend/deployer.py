"""Deployment orchestrator — deploys fine-tuned models on RunPod with vLLM."""

from __future__ import annotations

import json

from loguru import logger

from backend.config import Settings
from backend.runpod_client import create_deployment_pod, get_pod_status, terminate_pod
from backend.storage import create_job, get_latest_job, update_job, update_project_stage


async def deploy_model(project_id: str, settings: Settings, gpu_type: str | None = None) -> dict:
    """Deploy the fine-tuned model on RunPod using vLLM."""
    train_job = await get_latest_job(project_id, "train")
    if not train_job or train_job.status != "completed":
        raise RuntimeError("Training must complete before deployment.")

    train_meta = json.loads(train_job.metadata_json or "{}")
    hf_username = settings.hf_username
    hf_model_repo = f"{hf_username}/{settings.huggingface.repo_prefix}-{project_id}"

    job = await create_job(
        project_id=project_id,
        job_type="deploy",
        metadata={
            "hf_model_repo": hf_model_repo,
            "base_model": train_meta.get("base_model", ""),
        },
    )

    pod = create_deployment_pod(
        settings=settings,
        project_id=project_id,
        hf_model_repo=hf_model_repo,
        gpu_type=gpu_type,
    )

    pod_id = pod.get("id", "")
    await update_job(job.id, status="running", metadata={"runpod_pod_id": pod_id})
    await update_project_stage(project_id, "deploying")

    return {
        "job_id": job.id,
        "runpod_pod_id": pod_id,
        "hf_model_repo": hf_model_repo,
        "status": "deploying",
    }


async def get_deployment_status(project_id: str, settings: Settings) -> dict:
    """Get deployment pod status and endpoint URL."""
    job = await get_latest_job(project_id, "deploy")
    if not job:
        return {"status": "no_deployment", "project_id": project_id}

    metadata = json.loads(job.metadata_json or "{}")
    pod_id = metadata.get("runpod_pod_id", "")

    result = {
        "job_id": job.id,
        "project_id": project_id,
        "status": job.status,
        "hf_model_repo": metadata.get("hf_model_repo", ""),
        "endpoint_url": "",
    }

    if pod_id:
        try:
            pod_info = get_pod_status(settings, pod_id)
            pod_status = pod_info.get("desiredStatus", "unknown")
            runtime = pod_info.get("runtime", {})

            # Extract the public endpoint URL
            if runtime and runtime.get("ports"):
                for port_info in runtime["ports"]:
                    if port_info.get("privatePort") == 8000:
                        ip = port_info.get("ip", "")
                        pub_port = port_info.get("publicPort", "")
                        if ip and pub_port:
                            result["endpoint_url"] = f"https://{pod_id}-8000.proxy.runpod.net"

            if pod_status == "RUNNING" and result["endpoint_url"]:
                await update_job(job.id, status="completed", metadata={"endpoint_url": result["endpoint_url"]})
                await update_project_stage(project_id, "deployed")
                result["status"] = "deployed"
            else:
                result["status"] = "starting"
                result["pod_status"] = pod_status

        except Exception as e:
            logger.error(f"Failed to fetch deployment pod status: {e}")

    return result


async def stop_deployment(project_id: str, settings: Settings) -> dict:
    """Terminate the deployment pod."""
    job = await get_latest_job(project_id, "deploy")
    if not job:
        return {"status": "no_deployment"}

    metadata = json.loads(job.metadata_json or "{}")
    pod_id = metadata.get("runpod_pod_id", "")

    if pod_id:
        terminate_pod(settings, pod_id)

    await update_job(job.id, status="cancelled")
    return {"status": "stopped", "job_id": job.id}


def generate_sample_code(endpoint_url: str, model_name: str) -> dict:
    """Generate sample code snippets for using the deployed model."""
    return {
        "python_openai": f'''from openai import OpenAI

client = OpenAI(
    base_url="{endpoint_url}/v1",
    api_key="your-runpod-api-key",
)

response = client.chat.completions.create(
    model="{model_name}",
    messages=[
        {{"role": "system", "content": "You are a helpful assistant."}},
        {{"role": "user", "content": "Your question here"}},
    ],
    max_tokens=512,
    temperature=0.7,
)

print(response.choices[0].message.content)
''',

        "python_requests": f'''import requests

url = "{endpoint_url}/v1/chat/completions"
headers = {{
    "Authorization": "Bearer your-runpod-api-key",
    "Content-Type": "application/json",
}}
payload = {{
    "model": "{model_name}",
    "messages": [
        {{"role": "system", "content": "You are a helpful assistant."}},
        {{"role": "user", "content": "Your question here"}},
    ],
    "max_tokens": 512,
    "temperature": 0.7,
}}

response = requests.post(url, json=payload, headers=headers)
print(response.json()["choices"][0]["message"]["content"])
''',

        "curl": f'''curl -X POST {endpoint_url}/v1/chat/completions \\
  -H "Authorization: Bearer your-runpod-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{model_name}",
    "messages": [
      {{"role": "system", "content": "You are a helpful assistant."}},
      {{"role": "user", "content": "Your question here"}}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }}'
''',

        "javascript": f'''const response = await fetch("{endpoint_url}/v1/chat/completions", {{
  method: "POST",
  headers: {{
    "Authorization": "Bearer your-runpod-api-key",
    "Content-Type": "application/json",
  }},
  body: JSON.stringify({{
    model: "{model_name}",
    messages: [
      {{ role: "system", content: "You are a helpful assistant." }},
      {{ role: "user", content: "Your question here" }},
    ],
    max_tokens: 512,
    temperature: 0.7,
  }}),
}});

const data = await response.json();
console.log(data.choices[0].message.content);
''',
    }
