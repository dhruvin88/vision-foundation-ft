"""Modal cloud training for detection experiments.

SETUP (one-time):
    pip install modal
    modal setup          # authenticates with your account

WORKFLOW:
    # 1. Upload your local datasets to the Modal volume (run once per dataset)
    modal run experiments/modal_train.py::upload_datasets

    # 2. Run the default experiment: vitb14 + rtdetr on coco_15cls + coco_20cls
    modal run experiments/modal_train.py

    # 3. Run all decoder combos (rtdetr vs detr_multiscale, both encoders)
    modal run experiments/modal_train.py -- --all-combos

    # 4. Customise epochs / batch size
    modal run experiments/modal_train.py -- --epochs 100 --batch-size 8

    # 5. Download results back to local machine
    modal volume get fft-results / ./experiments/results_modal

GPU cost estimate (A10G @ ~$0.90/hr):
    default (2 runs × 50 epochs, parallel):    ~1.5 h  →  ~$1.50
    --all-combos (8 runs × 50 ep, parallel):   ~2.5 h  →  ~$9
"""

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

# ── Persistent volumes ────────────────────────────────────────────────────────
dataset_vol = modal.Volume.from_name("fft-datasets",  create_if_missing=True)
results_vol = modal.Volume.from_name("fft-results",   create_if_missing=True)
hub_vol     = modal.Volume.from_name("fft-hub-cache", create_if_missing=True)

# ── Container image ───────────────────────────────────────────────────────────
# In Modal 1.x, local code is baked into the image via add_local_dir().
# Heavy directories (venv, datasets, results, caches) are excluded via ignore=.
_SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", ".mypy_cache", "checkpoints"}
_SKIP_PATHS = {"experiments/datasets", "experiments/results", "experiments/results_modal"}

def _ignore(path: Path) -> bool:
    """Return True to exclude a path from the image."""
    parts = set(path.parts)
    if parts & _SKIP_DIRS:
        return True
    path_str = path.as_posix()
    return any(skip in path_str for skip in _SKIP_PATHS)

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
        add_python="3.11",
    )
    .pip_install(
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics[detection]>=1.0",
        "albumentations>=1.3.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0,<2.0.0",  # PyTorch 2.1.0 was built against NumPy 1.x
        "pycocotools>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "transformers>=4.50.0",
        "scipy>=1.10",
    )
    .add_local_dir(ROOT, remote_path="/app", ignore=_ignore)
)

# ── Modal app ─────────────────────────────────────────────────────────────────
app = modal.App("fft-detection")


@app.function(
    gpu="A10G",             # 24 GB VRAM — plenty for RTDETRDecoder + batch 16
    image=image,
    volumes={
        "/datasets":           dataset_vol,
        "/results":            results_vol,
        "/root/.cache/torch":  hub_vol,   # DINOv2 weights cached across runs
    },
    timeout=60 * 60 * 10,  # 10-hour hard cap
)
def run_experiment(
    dataset_name: str,
    encoder_name: str,
    decoder_type: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> dict:
    """Run a single detection experiment on a Modal GPU container."""
    import time
    from datetime import datetime

    sys.path.insert(0, "/app")

    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.training.trainer import Trainer
    from core.cli import _create_decoder

    dataset_path = Path("/datasets") / dataset_name
    if not (dataset_path / "annotations.json").exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in the Modal volume.\n"
            f"Run:  modal run experiments/modal_train.py::upload_datasets"
        )

    with open(dataset_path / "annotations.json") as f:
        num_classes = len(json.load(f)["categories"])

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_name}  ({num_classes} classes)")
    print(f"  Encoder : {encoder_name}")
    print(f"  Decoder : {decoder_type}")
    print(f"  Epochs  : {epochs}   Batch: {batch_size}   LR: {lr}")
    print(f"{'='*60}\n")

    t0 = time.time()

    encoder = create_encoder(encoder_name)
    if decoder_type in ["detr_multiscale", "fpn"]:
        encoder.intermediate_layers = encoder.default_intermediate_layers()

    decoder = _create_decoder(decoder_type, "detection", encoder, num_classes)
    print(f"Decoder  : {type(decoder).__name__}  ({decoder.num_trainable_params():,} params)")

    dataset = FFTDataset.from_folder(
        dataset_path, task="detection", transform=encoder.get_transform()
    )
    print(f"Samples  : {len(dataset)}")

    output_dir = Path("/results") / dataset_name / encoder_name / decoder_type
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        scheduler="cosine",
        warmup_epochs=5,
        augmentation="light",
        early_stopping_patience=20,
        checkpoint_dir=output_dir / "checkpoints",
        num_workers=4,
        accelerator="gpu",
    )

    results = trainer.fit()
    train_time = time.time() - t0

    results_dict = {
        "dataset":                dataset_name,
        "encoder":                encoder_name,
        "decoder":                decoder_type,
        "num_classes":            num_classes,
        "num_train_samples":      len(dataset),
        "num_params":             decoder.num_trainable_params(),
        "epochs_trained":         results["epochs_trained"],
        "batch_size":             batch_size,
        "lr":                     lr,
        "train_time_s":           train_time,
        "train_time_per_epoch_s": train_time / max(results["epochs_trained"], 1),
        **results,
        "timestamp":              datetime.now().isoformat(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    trainer.save(output_dir / "decoder_weights.pt")
    results_vol.commit()  # flush writes so they're visible immediately

    map50 = results.get("val_map50", float("nan"))
    mmap  = results.get("val_map",   float("nan"))
    print(
        f"\nDone — mAP@50={map50:.4f}  mAP@50:95={mmap:.4f}"
        f"  time={train_time / 3600:.2f}h"
    )
    return results_dict


# ── Local entrypoints ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    all_combos: bool = False,
    epochs: int = 50,
    batch_size: int = 16,
):
    """Submit detection experiments to Modal (runs in parallel)."""
    datasets = ["retail_coco", "coco_15cls", "coco_20cls"]

    if all_combos:
        pairs = [
            ("dinov2_vits14", "rtdetr"),
            ("dinov2_vits14", "detr_multiscale"),
            ("dinov2_vitb14", "rtdetr"),
            ("dinov2_vitb14", "detr_multiscale"),
        ]
    else:
        pairs = [
            ("dinov2_vitb14", "rtdetr"),
        ]

    runs = [
        (ds, enc, dec, epochs, batch_size, 1e-4)
        for ds in datasets
        for enc, dec in pairs
    ]

    total = len(runs)
    print(f"\nSubmitting {total} experiment(s) — Modal runs them in parallel.\n")
    for ds, enc, dec, ep, bs, _ in runs:
        print(f"  {ds:<16}  {enc:<18}  {dec:<18}  ep={ep}  bs={bs}")

    all_results = list(run_experiment.starmap(runs))

    print("\n\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(
        f"{'Dataset':<16} {'Encoder':<16} {'Decoder':<18}"
        f" {'mAP@50':>8} {'mAP:95':>8} {'Ep':>4} {'Time':>7}"
    )
    print("-" * 72)
    for r in all_results:
        map50 = r.get("val_map50", float("nan"))
        mmap  = r.get("val_map",   float("nan"))
        h     = r.get("train_time_s", 0) / 3600
        print(
            f"{r['dataset']:<16} {r['encoder']:<16} {r['decoder']:<18}"
            f" {map50:>8.4f} {mmap:>8.4f} {r['epochs_trained']:>4} {h:>6.2f}h"
        )

    print(f"\nResults saved to Modal volume 'fft-results'.")
    print("Download with:  modal volume get fft-results / ./experiments/results_modal")


@app.local_entrypoint()
def upload_datasets():
    """Upload local datasets to the 'fft-datasets' Modal volume.

    Run once (or again whenever you add a new dataset):
        modal run experiments/modal_train.py::upload_datasets
    """
    datasets_dir = ROOT / "experiments" / "datasets"
    if not datasets_dir.exists():
        print(f"No datasets directory found at {datasets_dir}")
        return

    # Only upload folders that have annotations.json (prepared detection datasets)
    subdirs = [
        d for d in datasets_dir.iterdir()
        if d.is_dir() and (d / "annotations.json").exists()
    ]

    if not subdirs:
        print("No prepared detection datasets found — run prepare_detection_datasets.py first.")
        return

    print(f"Uploading {len(subdirs)} dataset(s) to Modal volume 'fft-datasets'...")
    with dataset_vol.batch_upload(force=True) as batch:
        for ds_dir in subdirs:
            print(f"  {ds_dir.name}/ ...")
            batch.put_directory(str(ds_dir), f"/{ds_dir.name}")

    print("\nUpload complete. Verify with:  modal volume ls fft-datasets")
