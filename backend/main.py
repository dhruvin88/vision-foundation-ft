"""FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import datasets, models, projects, training
from backend.db.database import create_db_and_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Foundation Model Fine-Tuning Platform",
    description="Fine-tune lightweight heads on frozen vision foundation models",
    version="0.1.0",
)

# CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(models.router)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/api/encoders")
def list_encoders():
    """List available encoder models."""
    from core.encoders import ALL_ENCODER_VARIANTS
    return {
        name: {
            "embed_dim": config["embed_dim"],
            "patch_size": config["patch_size"],
            "num_heads": config["num_heads"],
        }
        for name, config in ALL_ENCODER_VARIANTS.items()
    }


@app.get("/api/decoders")
def list_decoders():
    """List available decoder heads."""
    return {
        "classification": [
            {"name": "linear_probe", "description": "Single linear layer (simplest)"},
            {"name": "mlp", "description": "2-layer MLP with dropout"},
            {"name": "transformer", "description": "Cross-attention transformer head"},
        ],
        "detection": [
            {"name": "detr_lite", "description": "Lightweight DETR-style decoder"},
            {"name": "fpn", "description": "Feature Pyramid Network + anchors"},
        ],
        "segmentation": [
            {"name": "linear_seg", "description": "Per-patch linear classifier"},
            {"name": "upernet", "description": "UPerNet multi-scale decoder"},
            {"name": "mask_transformer", "description": "Mask prediction via dot-product"},
        ],
    }
