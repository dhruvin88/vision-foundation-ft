"""Training job orchestration service."""

from __future__ import annotations

import asyncio
import datetime
import logging
import traceback
from typing import Any

from sqlmodel import Session

from backend.db.models import TrainingRun
from backend.services.storage import get_images_dir, get_models_dir

logger = logging.getLogger(__name__)

# Track running jobs
_active_jobs: dict[int, asyncio.Task] = {}
_job_progress: dict[int, dict[str, Any]] = {}


async def start_training_job(run_id: int, session_factory) -> None:
    """Start a training job in the background.

    Args:
        run_id: ID of the TrainingRun record.
        session_factory: Callable that returns a database session.
    """
    if run_id in _active_jobs:
        raise RuntimeError(f"Training run {run_id} is already active")

    task = asyncio.create_task(_run_training(run_id, session_factory))
    _active_jobs[run_id] = task
    _job_progress[run_id] = {"status": "starting", "epoch": 0, "metrics": {}}


async def cancel_training_job(run_id: int) -> None:
    """Cancel a running training job."""
    if run_id in _active_jobs:
        _active_jobs[run_id].cancel()
        del _active_jobs[run_id]
        _job_progress[run_id] = {"status": "cancelled"}
        logger.info("Cancelled training run %d", run_id)


def get_job_progress(run_id: int) -> dict[str, Any]:
    """Get the current progress of a training job."""
    return _job_progress.get(run_id, {"status": "unknown"})


async def _run_training(run_id: int, session_factory) -> None:
    """Execute the training loop in a background task."""
    try:
        # Update status to running
        with session_factory() as session:
            run = session.get(TrainingRun, run_id)
            if run is None:
                logger.error("TrainingRun %d not found", run_id)
                return
            run.status = "running"
            run.started_at = datetime.datetime.utcnow()
            session.add(run)
            session.commit()

            # Capture config before closing session
            config = {
                "project_id": run.project_id,
                "encoder_name": run.encoder_name,
                "decoder_name": run.decoder_name,
                "num_classes": run.num_classes,
                "learning_rate": run.learning_rate,
                "epochs": run.epochs,
                "batch_size": run.batch_size,
                "scheduler": run.scheduler,
                "augmentation": run.augmentation,
            }

        _job_progress[run_id] = {"status": "loading_encoder", "epoch": 0, "metrics": {}}

        # Run training in a thread to avoid blocking the event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None, _train_sync, run_id, config
        )

        # Update with results
        with session_factory() as session:
            run = session.get(TrainingRun, run_id)
            if run is not None:
                run.status = "completed"
                run.completed_at = datetime.datetime.utcnow()
                run.best_metric = result.get("best_val_loss", 0)
                run.weights_path = result.get("weights_path", "")
                session.add(run)
                session.commit()

        _job_progress[run_id] = {"status": "completed", "metrics": result}

    except asyncio.CancelledError:
        with session_factory() as session:
            run = session.get(TrainingRun, run_id)
            if run is not None:
                run.status = "cancelled"
                session.add(run)
                session.commit()
        raise

    except Exception as e:
        logger.error("Training run %d failed: %s", run_id, traceback.format_exc())
        with session_factory() as session:
            run = session.get(TrainingRun, run_id)
            if run is not None:
                run.status = "failed"
                run.error_message = str(e)
                session.add(run)
                session.commit()
        _job_progress[run_id] = {"status": "failed", "error": str(e)}

    finally:
        _active_jobs.pop(run_id, None)


def _train_sync(run_id: int, config: dict) -> dict:
    """Synchronous training function (runs in executor)."""
    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.training.trainer import Trainer
    from core.cli import _create_decoder

    project_id = config["project_id"]

    # Load encoder
    _job_progress[run_id]["status"] = "loading_encoder"
    encoder = create_encoder(config["encoder_name"])

    # Create decoder
    decoder = _create_decoder(
        config["decoder_name"], "classification", encoder, config["num_classes"]
    )

    # Load dataset from project images
    images_dir = get_images_dir(project_id)
    dataset = FFTDataset.from_folder(
        images_dir.parent, task="classification", transform=encoder.get_transform()
    )

    # Train
    models_dir = get_models_dir(project_id)
    trainer = Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=config["learning_rate"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        scheduler=config["scheduler"],
        augmentation=config["augmentation"],
        checkpoint_dir=models_dir / "checkpoints",
    )

    _job_progress[run_id]["status"] = "training"
    results = trainer.fit()

    # Save weights
    weights_path = models_dir / f"run_{run_id}_weights.pt"
    trainer.save(weights_path)
    results["weights_path"] = str(weights_path)

    return results
