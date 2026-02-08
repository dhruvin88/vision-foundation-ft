"""Training management API endpoints."""

from __future__ import annotations

import asyncio
import datetime

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.db.database import get_session
from backend.db.models import Project, TrainingRun
from backend.services import job_runner, websocket as ws_service

router = APIRouter(prefix="/api/projects/{project_id}/training", tags=["training"])


class TrainingConfig(BaseModel):
    encoder_name: str = "dinov2_vitb14"
    decoder_name: str = "auto"
    num_classes: int = 2
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    scheduler: str = "cosine"
    augmentation: str = "light"


class TrainingRunResponse(BaseModel):
    id: int
    project_id: int
    status: str
    encoder_name: str
    decoder_name: str
    num_classes: int
    learning_rate: float
    epochs: int
    batch_size: int
    scheduler: str
    augmentation: str
    best_metric: float
    weights_path: str
    error_message: str
    started_at: datetime.datetime | None
    completed_at: datetime.datetime | None
    created_at: datetime.datetime


@router.post("/start", response_model=TrainingRunResponse)
async def start_training(
    project_id: int,
    config: TrainingConfig,
    session: Session = Depends(get_session),
):
    """Start a new training run."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate encoder name
    from core.encoders.dinov2 import ALL_VARIANTS
    if config.encoder_name not in ALL_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown encoder: {config.encoder_name}",
        )

    run = TrainingRun(
        project_id=project_id,
        encoder_name=config.encoder_name,
        decoder_name=config.decoder_name,
        num_classes=config.num_classes,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        batch_size=config.batch_size,
        scheduler=config.scheduler,
        augmentation=config.augmentation,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    # Start training in background
    from backend.db.database import Session as SessionClass, engine

    def session_factory():
        return SessionClass(engine)

    await job_runner.start_training_job(run.id, session_factory)

    return run


@router.get("/runs", response_model=list[TrainingRunResponse])
def list_training_runs(
    project_id: int, session: Session = Depends(get_session)
):
    """List all training runs for a project."""
    runs = session.exec(
        select(TrainingRun)
        .where(TrainingRun.project_id == project_id)
        .order_by(TrainingRun.created_at.desc())
    ).all()
    return runs


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
def get_training_run(
    project_id: int, run_id: int, session: Session = Depends(get_session)
):
    """Get a specific training run."""
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run


@router.post("/runs/{run_id}/cancel")
async def cancel_training(
    project_id: int, run_id: int, session: Session = Depends(get_session)
):
    """Cancel a running training job."""
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status != "running":
        raise HTTPException(status_code=400, detail="Training run is not running")

    await job_runner.cancel_training_job(run_id)

    run.status = "cancelled"
    session.add(run)
    session.commit()
    return {"status": "cancelled"}


@router.get("/runs/{run_id}/progress")
def get_progress(project_id: int, run_id: int):
    """Get current training progress (poll-based alternative to WebSocket)."""
    return job_runner.get_job_progress(run_id)


@router.websocket("/runs/{run_id}/ws")
async def training_websocket(
    websocket: WebSocket, project_id: int, run_id: int
):
    """WebSocket endpoint for real-time training progress."""
    await ws_service.connect(run_id, websocket)

    # Start progress monitor
    monitor_task = asyncio.create_task(ws_service.progress_monitor(run_id))

    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        monitor_task.cancel()
        await ws_service.disconnect(run_id, websocket)
