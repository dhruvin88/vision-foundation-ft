"""SQLModel database models for the backend."""

from __future__ import annotations

import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Project(SQLModel, table=True):
    """A user project containing a dataset and training runs."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str = ""
    task: str = Field(description="classification, detection, or segmentation")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class ImageRecord(SQLModel, table=True):
    """A single image in a project's dataset."""

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    filename: str
    original_filename: str
    file_path: str
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    split: str = Field(default="train", description="train or val")
    uploaded_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class Annotation(SQLModel, table=True):
    """An annotation for an image (label, bounding box, or mask)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="imagerecord.id", index=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    annotation_type: str = Field(description="class_label, bbox, or mask")
    class_name: str = ""
    class_id: int = 0
    # Bounding box fields (for detection)
    bbox_x: float = 0
    bbox_y: float = 0
    bbox_w: float = 0
    bbox_h: float = 0
    # Mask path (for segmentation)
    mask_path: str = ""
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class TrainingRun(SQLModel, table=True):
    """A training run for a project."""

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    status: str = Field(default="pending", description="pending, running, completed, failed, cancelled")
    encoder_name: str = "dinov2_vitb14"
    decoder_name: str = "auto"
    num_classes: int = 2
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    scheduler: str = "cosine"
    augmentation: str = "light"
    # Results
    best_metric: float = 0.0
    weights_path: str = ""
    error_message: str = ""
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
