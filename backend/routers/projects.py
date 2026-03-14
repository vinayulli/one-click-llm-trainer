from fastapi import APIRouter, HTTPException

from backend.models import ProjectCreate, ProjectResponse, ProjectStage
from backend.storage import create_project, get_project, list_projects

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse)
async def create_new_project(req: ProjectCreate):
    row = await create_project(name=req.name, description=req.description)
    return ProjectResponse(
        id=row.id,
        name=row.name,
        description=row.description,
        stage=ProjectStage(row.stage),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("", response_model=list[ProjectResponse])
async def get_all_projects():
    rows = await list_projects()
    return [
        ProjectResponse(
            id=r.id,
            name=r.name,
            description=r.description,
            stage=ProjectStage(r.stage),
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project_detail(project_id: str):
    row = await get_project(project_id)
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectResponse(
        id=row.id,
        name=row.name,
        description=row.description,
        stage=ProjectStage(row.stage),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
