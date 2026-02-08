"""Model management API endpoints (download, inference)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session

from backend.db.database import get_session
from backend.db.models import TrainingRun
from backend.services.storage import get_models_dir

router = APIRouter(prefix="/api/projects/{project_id}/models", tags=["models"])


class InferenceRequest(BaseModel):
    run_id: int


class InferenceResult(BaseModel):
    image_name: str
    predictions: dict


@router.get("/runs/{run_id}/download")
def download_weights(
    project_id: int, run_id: int, session: Session = Depends(get_session)
):
    """Download trained decoder weights."""
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")

    weights_path = Path(run.weights_path)
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights file not found")

    return FileResponse(
        path=str(weights_path),
        filename=f"decoder_weights_run_{run_id}.pt",
        media_type="application/octet-stream",
    )


@router.post("/runs/{run_id}/inference", response_model=list[InferenceResult])
async def run_inference(
    project_id: int,
    run_id: int,
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
):
    """Run inference on uploaded images using a trained model."""
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")

    weights_path = Path(run.weights_path)
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights file not found")

    # Save uploaded images temporarily
    import tempfile

    results = []
    temp_dir = Path(tempfile.mkdtemp())

    try:
        image_paths = []
        for file in files:
            temp_path = temp_dir / (file.filename or "image.jpg")
            content = await file.read()
            temp_path.write_bytes(content)
            image_paths.append(temp_path)

        # Load model and run inference
        from core.encoders.dinov2 import DINOv2Encoder
        from core.evaluation.inference import run_inference as _run_inference
        from core.export.weights import load_decoder_weights
        from core.cli import _create_decoder

        encoder = DINOv2Encoder(run.encoder_name)
        # Determine task from project
        from backend.db.models import Project
        project = session.get(Project, project_id)
        task = project.task if project else "classification"

        decoder = _create_decoder(run.decoder_name, task, encoder, run.num_classes)
        load_decoder_weights(decoder, weights_path)

        predictions = _run_inference(decoder, image_paths)

        for pred in predictions:
            image_name = Path(pred["image_path"]).name
            # Filter out non-serializable items
            pred_dict = {
                k: v for k, v in pred.items()
                if k != "image_path" and not hasattr(v, "shape")
            }
            results.append(
                InferenceResult(image_name=image_name, predictions=pred_dict)
            )

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


@router.get("/runs/{run_id}/script")
def get_inference_script(
    project_id: int, run_id: int, session: Session = Depends(get_session)
):
    """Generate and return a standalone inference script."""
    run = session.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(status_code=404, detail="Training run not found")

    from core.export.script_gen import generate_inference_script
    from backend.db.models import Project

    project = session.get(Project, project_id)
    task = project.task if project else "classification"

    models_dir = get_models_dir(project_id)
    script_path = models_dir / f"inference_run_{run_id}.py"

    # Map decoder_name to class name
    decoder_class_map = {
        "auto": "LinearProbe",
        "linear_probe": "LinearProbe",
        "mlp": "MLPHead",
        "transformer": "TransformerHead",
        "detr_lite": "DETRLiteDecoder",
        "fpn": "FPNHead",
        "linear_seg": "LinearSegHead",
        "upernet": "UPerNetHead",
        "mask_transformer": "MaskTransformerHead",
    }
    decoder_class = decoder_class_map.get(run.decoder_name, "LinearProbe")

    script = generate_inference_script(
        decoder_class=decoder_class,
        task=task,
        encoder_name=run.encoder_name,
        num_classes=run.num_classes,
        weights_path=run.weights_path,
        output_path=script_path,
    )

    return {"script": script, "download_url": f"/api/projects/{project_id}/models/runs/{run_id}/download"}
