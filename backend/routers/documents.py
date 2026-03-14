from fastapi import APIRouter, HTTPException, UploadFile, File
from loguru import logger

from backend.config import settings
from backend.document_processor import process_documents, save_uploaded_file
from backend.models import ProcessResponse, UploadResponse
from backend.storage import get_project, update_project_stage

router = APIRouter(prefix="/api/projects/{project_id}", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(project_id: str, files: list[UploadFile] = File(...)):
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    saved_files = []
    for f in files:
        content = await f.read()
        save_uploaded_file(content, f.filename, project_id, settings)
        saved_files.append(f.filename)
        logger.info(f"Uploaded: {f.filename}")

    await update_project_stage(project_id, "documents_uploaded")

    return UploadResponse(
        project_id=project_id,
        files_saved=saved_files,
        total_files=len(saved_files),
    )


@router.post("/process", response_model=ProcessResponse)
async def process_uploaded_documents(project_id: str):
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    result = process_documents(project_id, settings)

    await update_project_stage(project_id, "documents_processed")

    return ProcessResponse(
        project_id=project_id,
        files_processed=result["files_processed"],
        total_chunks=result["total_chunks"],
    )


@router.get("/documents")
async def list_documents(project_id: str):
    raw_dir = settings.project_raw_dir(project_id)
    if not raw_dir.exists():
        return {"project_id": project_id, "files": []}

    files = [
        {"name": f.name, "size_bytes": f.stat().st_size}
        for f in raw_dir.iterdir()
        if f.is_file()
    ]
    return {"project_id": project_id, "files": files}
