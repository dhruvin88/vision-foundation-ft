"""File storage service for images, models, and other artifacts."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "./uploads"))
THUMBNAIL_SIZE = (256, 256)


def get_project_dir(project_id: int) -> Path:
    """Get the storage directory for a project."""
    project_dir = STORAGE_DIR / f"project_{project_id}"
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def get_images_dir(project_id: int) -> Path:
    """Get the images directory for a project."""
    images_dir = get_project_dir(project_id) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def get_thumbnails_dir(project_id: int) -> Path:
    """Get the thumbnails directory for a project."""
    thumbs_dir = get_project_dir(project_id) / "thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    return thumbs_dir


def get_models_dir(project_id: int) -> Path:
    """Get the models directory for a project."""
    models_dir = get_project_dir(project_id) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


async def save_uploaded_image(
    project_id: int, file_content: bytes, original_filename: str
) -> dict:
    """Save an uploaded image and generate a thumbnail.

    Returns:
        Dict with file_path, filename, width, height, file_size_bytes.
    """
    # Generate unique filename
    ext = Path(original_filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}:
        raise ValueError(f"Unsupported image format: {ext}")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    images_dir = get_images_dir(project_id)
    file_path = images_dir / unique_name

    # Save image
    file_path.write_bytes(file_content)

    # Get image dimensions
    with Image.open(file_path) as img:
        width, height = img.size

    # Generate thumbnail
    thumbs_dir = get_thumbnails_dir(project_id)
    thumb_path = thumbs_dir / f"{unique_name}.jpg"
    with Image.open(file_path) as img:
        img.thumbnail(THUMBNAIL_SIZE)
        img.convert("RGB").save(thumb_path, "JPEG", quality=85)

    return {
        "file_path": str(file_path),
        "filename": unique_name,
        "width": width,
        "height": height,
        "file_size_bytes": len(file_content),
    }


def delete_project_files(project_id: int) -> None:
    """Delete all files for a project."""
    project_dir = get_project_dir(project_id)
    if project_dir.exists():
        shutil.rmtree(project_dir)
        logger.info("Deleted project files: %s", project_dir)
