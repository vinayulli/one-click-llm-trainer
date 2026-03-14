from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class ProjectRow(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    stage = Column(String, default="created")
    config_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class JobRow(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    project_id = Column(String, nullable=False, index=True)
    job_type = Column(String, nullable=False)  # train, evaluate, deploy
    runpod_pod_id = Column(String, default="")
    status = Column(String, default="pending")
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Database engine
# ---------------------------------------------------------------------------

engine = create_async_engine(settings.db_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

async def create_project(name: str, description: str = "") -> ProjectRow:
    row = ProjectRow(
        id=uuid.uuid4().hex[:12],
        name=name,
        description=description,
    )
    async with async_session() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def get_project(project_id: str) -> ProjectRow | None:
    async with async_session() as session:
        return await session.get(ProjectRow, project_id)


async def list_projects() -> list[ProjectRow]:
    from sqlalchemy import select

    async with async_session() as session:
        result = await session.execute(
            select(ProjectRow).order_by(ProjectRow.created_at.desc())
        )
        return list(result.scalars().all())


async def update_project_stage(project_id: str, stage: str):
    async with async_session() as session:
        row = await session.get(ProjectRow, project_id)
        if row:
            row.stage = stage
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()


async def update_project_config(project_id: str, config: dict):
    async with async_session() as session:
        row = await session.get(ProjectRow, project_id)
        if row:
            row.config_json = json.dumps(config)
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()


async def create_job(
    project_id: str,
    job_type: str,
    runpod_pod_id: str = "",
    metadata: dict | None = None,
) -> JobRow:
    row = JobRow(
        id=uuid.uuid4().hex[:12],
        project_id=project_id,
        job_type=job_type,
        runpod_pod_id=runpod_pod_id,
        metadata_json=json.dumps(metadata or {}),
    )
    async with async_session() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def get_latest_job(project_id: str, job_type: str) -> JobRow | None:
    from sqlalchemy import select

    async with async_session() as session:
        result = await session.execute(
            select(JobRow)
            .where(JobRow.project_id == project_id, JobRow.job_type == job_type)
            .order_by(JobRow.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


async def update_job(job_id: str, status: str | None = None, metadata: dict | None = None):
    async with async_session() as session:
        row = await session.get(JobRow, job_id)
        if row:
            if status:
                row.status = status
            if metadata:
                existing = json.loads(row.metadata_json or "{}")
                existing.update(metadata)
                row.metadata_json = json.dumps(existing)
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()


async def get_active_jobs() -> list[JobRow]:
    from sqlalchemy import select

    async with async_session() as session:
        result = await session.execute(
            select(JobRow).where(JobRow.status.in_(["pending", "running"]))
        )
        return list(result.scalars().all())
