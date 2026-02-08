"""Dataset management API endpoints (upload, annotate, organize)."""

from __future__ import annotations

import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.db.database import get_session
from backend.db.models import Annotation, ImageRecord, Project
from backend.services.storage import save_uploaded_image

router = APIRouter(prefix="/api/projects/{project_id}/dataset", tags=["datasets"])

MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB


class AnnotationCreate(BaseModel):
    annotation_type: str  # class_label, bbox, mask
    class_name: str = ""
    class_id: int = 0
    bbox_x: float = 0
    bbox_y: float = 0
    bbox_w: float = 0
    bbox_h: float = 0
    mask_path: str = ""


class ImageResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    width: int
    height: int
    file_size_bytes: int
    split: str
    uploaded_at: datetime.datetime


class AnnotationResponse(BaseModel):
    id: int
    image_id: int
    annotation_type: str
    class_name: str
    class_id: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float


class DatasetStatsResponse(BaseModel):
    total_images: int
    train_images: int
    val_images: int
    total_annotations: int
    class_distribution: dict[str, int]


def _get_project(project_id: int, session: Session) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/upload", response_model=list[ImageResponse])
async def upload_images(
    project_id: int,
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
):
    """Upload one or more images to the project dataset."""
    _get_project(project_id, session)

    results = []
    for file in files:
        if file.size and file.size > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds maximum size of 50MB",
            )

        content = await file.read()
        file_info = await save_uploaded_image(
            project_id, content, file.filename or "unknown.jpg"
        )

        record = ImageRecord(
            project_id=project_id,
            filename=file_info["filename"],
            original_filename=file.filename or "unknown.jpg",
            file_path=file_info["file_path"],
            width=file_info["width"],
            height=file_info["height"],
            file_size_bytes=file_info["file_size_bytes"],
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        results.append(record)

    return results


@router.get("/images", response_model=list[ImageResponse])
def list_images(
    project_id: int,
    split: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """List all images in the dataset, optionally filtered by split."""
    _get_project(project_id, session)

    query = select(ImageRecord).where(ImageRecord.project_id == project_id)
    if split:
        query = query.where(ImageRecord.split == split)
    query = query.order_by(ImageRecord.uploaded_at.desc())

    images = session.exec(query).all()
    return images


@router.delete("/images/{image_id}")
def delete_image(
    project_id: int, image_id: int, session: Session = Depends(get_session)
):
    """Delete an image and its annotations."""
    image = session.get(ImageRecord, image_id)
    if not image or image.project_id != project_id:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete annotations
    annotations = session.exec(
        select(Annotation).where(Annotation.image_id == image_id)
    ).all()
    for ann in annotations:
        session.delete(ann)

    session.delete(image)
    session.commit()
    return {"status": "deleted"}


@router.post("/images/{image_id}/annotate", response_model=AnnotationResponse)
def annotate_image(
    project_id: int,
    image_id: int,
    data: AnnotationCreate,
    session: Session = Depends(get_session),
):
    """Add an annotation to an image."""
    image = session.get(ImageRecord, image_id)
    if not image or image.project_id != project_id:
        raise HTTPException(status_code=404, detail="Image not found")

    annotation = Annotation(
        image_id=image_id,
        project_id=project_id,
        annotation_type=data.annotation_type,
        class_name=data.class_name,
        class_id=data.class_id,
        bbox_x=data.bbox_x,
        bbox_y=data.bbox_y,
        bbox_w=data.bbox_w,
        bbox_h=data.bbox_h,
        mask_path=data.mask_path,
    )
    session.add(annotation)
    session.commit()
    session.refresh(annotation)
    return annotation


@router.get("/images/{image_id}/annotations", response_model=list[AnnotationResponse])
def get_annotations(
    project_id: int, image_id: int, session: Session = Depends(get_session)
):
    """Get all annotations for an image."""
    image = session.get(ImageRecord, image_id)
    if not image or image.project_id != project_id:
        raise HTTPException(status_code=404, detail="Image not found")

    annotations = session.exec(
        select(Annotation).where(Annotation.image_id == image_id)
    ).all()
    return annotations


@router.put("/images/{image_id}/split")
def set_image_split(
    project_id: int,
    image_id: int,
    split: str,
    session: Session = Depends(get_session),
):
    """Set the train/val split for an image."""
    if split not in ("train", "val"):
        raise HTTPException(status_code=400, detail="Split must be 'train' or 'val'")

    image = session.get(ImageRecord, image_id)
    if not image or image.project_id != project_id:
        raise HTTPException(status_code=404, detail="Image not found")

    image.split = split
    session.add(image)
    session.commit()
    return {"status": "updated", "split": split}


@router.get("/stats", response_model=DatasetStatsResponse)
def get_dataset_stats(project_id: int, session: Session = Depends(get_session)):
    """Get dataset statistics."""
    _get_project(project_id, session)

    images = session.exec(
        select(ImageRecord).where(ImageRecord.project_id == project_id)
    ).all()

    annotations = session.exec(
        select(Annotation).where(Annotation.project_id == project_id)
    ).all()

    train_count = sum(1 for img in images if img.split == "train")
    val_count = sum(1 for img in images if img.split == "val")

    class_dist: dict[str, int] = {}
    for ann in annotations:
        name = ann.class_name or f"class_{ann.class_id}"
        class_dist[name] = class_dist.get(name, 0) + 1

    return DatasetStatsResponse(
        total_images=len(images),
        train_images=train_count,
        val_images=val_count,
        total_annotations=len(annotations),
        class_distribution=class_dist,
    )
