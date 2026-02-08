"""Project management API endpoints."""

from __future__ import annotations

import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.db.database import get_session
from backend.db.models import Project
from backend.services.storage import delete_project_files

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    task: str  # classification, detection, segmentation


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str
    task: str
    created_at: datetime.datetime
    updated_at: datetime.datetime


@router.post("/", response_model=ProjectResponse)
def create_project(data: ProjectCreate, session: Session = Depends(get_session)):
    """Create a new project."""
    if data.task not in ("classification", "detection", "segmentation"):
        raise HTTPException(status_code=400, detail=f"Invalid task: {data.task}")

    project = Project(name=data.name, description=data.description, task=data.task)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.get("/", response_model=list[ProjectResponse])
def list_projects(session: Session = Depends(get_session)):
    """List all projects."""
    projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
    return projects


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, session: Session = Depends(get_session)):
    """Get a single project."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: int, data: ProjectCreate, session: Session = Depends(get_session)
):
    """Update a project."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.name = data.name
    project.description = data.description
    project.task = data.task
    project.updated_at = datetime.datetime.utcnow()
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.delete("/{project_id}")
def delete_project(project_id: int, session: Session = Depends(get_session)):
    """Delete a project and all its files."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    delete_project_files(project_id)
    session.delete(project)
    session.commit()
    return {"status": "deleted"}
