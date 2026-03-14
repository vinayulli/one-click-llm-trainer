"""Document upload, text extraction, and chunking."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pypdf import PdfReader

from backend.config import Settings


def extract_text_from_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_path: Path) -> str:
    from docx import Document
    doc = Document(str(file_path))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text_from_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
}


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if not extractor:
        raise ValueError(f"Unsupported file format: {suffix}")
    return extractor(file_path)


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def save_uploaded_file(file_content: bytes, filename: str, project_id: str, settings: Settings) -> Path:
    raw_dir = settings.project_raw_dir(project_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename
    dest.write_bytes(file_content)
    logger.info(f"Saved uploaded file: {dest}")
    return dest


def process_documents(project_id: str, settings: Settings) -> dict:
    raw_dir = settings.project_raw_dir(project_id)
    processed_dir = settings.project_processed_dir(project_id)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cfg = settings.document_processing
    all_chunks: list[dict] = []
    processed_files: list[str] = []

    if not raw_dir.exists():
        raise FileNotFoundError(f"No uploaded documents found for project {project_id}")

    for file_path in raw_dir.iterdir():
        if file_path.suffix.lower() not in cfg.supported_formats:
            logger.warning(f"Skipping unsupported file: {file_path.name}")
            continue

        logger.info(f"Processing: {file_path.name}")
        text = extract_text(file_path)
        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)

        for i, chunk_text_str in enumerate(chunks):
            all_chunks.append({
                "source": file_path.name,
                "chunk_id": i,
                "text": chunk_text_str,
            })
        processed_files.append(file_path.name)

    chunks_path = processed_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Processed {len(processed_files)} files into {len(all_chunks)} chunks")
    return {
        "files_processed": processed_files,
        "total_chunks": len(all_chunks),
        "chunks_path": str(chunks_path),
    }
