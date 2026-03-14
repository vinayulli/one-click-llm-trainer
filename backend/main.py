"""One Click LLM Trainer — FastAPI Application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.exceptions import OCLTError
from backend.scheduler import start_scheduler, stop_scheduler
from backend.storage import init_db

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting One Click LLM Trainer")
    await init_db()
    start_scheduler()
    yield
    stop_scheduler()
    logger.info("Shutting down")


app = FastAPI(
    title="One Click LLM Trainer",
    description="Upload docs → Generate dataset → Auto-select model → Fine-tune on RunPod → Evaluate → Deploy",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(OCLTError)
async def handle_oclt_error(request: Request, exc: OCLTError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


# --- Register routers ---
from backend.routers import datasets, deployment, documents, evaluation, projects, training

app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(evaluation.router)
app.include_router(deployment.router)


# --- Serve frontend ---
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}
