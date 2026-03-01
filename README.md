# Foundation Model Fine-Tuning (FFT)

> Fine-tune lightweight decoder heads on frozen vision foundation models for classification, object detection, and semantic segmentation.

## Table of Contents

- [Purpose](#purpose)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Key Concepts](#key-concepts)

---

## Purpose

FFT lets you adapt large pretrained vision models (DINOv2, DINOv3) to new tasks without modifying the encoder weights. Only a small task-specific head is trained, making the process fast and memory-efficient.

**Primary use cases:**
- Image classification on a custom dataset
- Object detection with RT-DETR or DETR-style heads
- Semantic segmentation with UPerNet or linear heads
- LoRA-based partial encoder fine-tuning when frozen features are insufficient

**Intended audience:** ML practitioners and researchers who want strong vision baselines quickly.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Input Image                    │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│         Frozen Encoder (DINOv3 / DINOv2)         │
│  DINOv3: vits16 · vitb16 · vitl16 (patch 16)    │
│  DINOv2: vits14 · vitb14 · vitl14 · vitg14      │
│  Optional: LoRA adapters on attention layers     │
└──────────────────────┬──────────────────────────┘
                       │  feature dict
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌─────────────┐ ┌────────────┐ ┌──────────────┐
│Classification│ │ Detection  │ │ Segmentation │
├─────────────┤ ├────────────┤ ├──────────────┤
│ LinearProbe │ │ RTDETRDec. │ │ LinearSeg    │
│ MLPHead     │ │ DETRLite   │ │ UPerNet      │
│ Transformer │ │ FPNHead    │ │ MaskTransf.  │
└─────────────┘ └────────────┘ └──────────────┘
```

The encoder runs in `torch.no_grad()` (or via LoRA gradients when enabled). Only the decoder parameters receive gradients, so GPU memory is dominated by the decoder, not the 300M–1.5B parameter encoder.

### System Components

```
Interfaces
  CLI (fft train / predict / info)
  Python SDK (import sdk as fft)
  REST API (FastAPI, uvicorn)
  Streamlit UI (streamlit run frontend/app.py)
           │
    Core Library
  ┌──────────┬──────────┬──────────┬────────┐
  │ Encoders │ Decoders │ Training │  Data  │
  └──────────┴──────────┴──────────┴────────┘
  ┌────────────┬────────┐
  │ Evaluation │ Export │
  └────────────┴────────┘
```

---

## Project Structure

```
core/
  encoders/          # DINOv2Encoder, DINOv3Encoder, LoRA injection
  decoders/          # RTDETRDecoder, DETRLite, FPNHead, classification, segmentation heads
  data/              # FFTDataset, format converters (COCO/VOC/YOLO/CSV), augmentations
  training/          # PyTorch Lightning Trainer, schedulers, callbacks
  evaluation/        # run_inference(), compute_metrics()
  export/            # save/load decoder weights, inference script generation
  cli.py             # fft CLI entry point
sdk/
  __init__.py        # High-level SDK: Encoder, ClassificationHead, DetectionHead, etc.
backend/
  main.py            # FastAPI application
  api/               # Route handlers (projects, datasets, training, models)
  db/                # SQLModel database setup
  services/          # storage, job_runner, websocket
frontend/
  app.py             # Streamlit entry point
  pages/             # 1_Dataset, 2_Model, 3_Training, 4_Results, 5_Inference
experiments/
  prepare_oxford_pets.py  # Download Oxford-IIIT Pets, convert masks to COCO boxes
  train_deim_pets.py      # RTDETRDecoder training on Oxford Pets (37 classes)
  datasets/               # Local dataset storage (gitignored)
  results/                # Training results (gitignored)
examples/
  classification_example.py
  detection_example.py / detection_e2e.py
  segmentation_example.py
tests/
  test_api/          # FastAPI endpoint tests
  test_decoders/     # Classification, detection, segmentation decoder tests
  test_encoders/     # DINOv2 and DINOv3 encoder tests
  test_training/     # Trainer and scheduler tests
docs/
  architecture.md    # Component deep-dives and data flow diagrams
  getting-started.md # Step-by-step tutorials
  sdk-reference.md   # Full SDK API reference
  rtdetr.md          # RTDETRDecoder architecture guide
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- GPU recommended but not required (CPU training works, expect slower speed)
- On this repo's dev machine: conda env `fft` at `~/.conda/envs/fft`

### Installation

```bash
git clone https://github.com/your-org/foundation-model-any-ft.git
cd foundation-model-any-ft

# Core library only
pip install -e .

# Core + REST API backend
pip install -e ".[backend]"

# Core + dev tools (pytest, ruff)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

The `transformers` package is a core dependency (required for DINOv3). DINOv2 weights are fetched from `torch.hub` (`facebookresearch/dinov2`). Both download automatically on first use.

### Running the Application

**REST API backend:**
```bash
uvicorn backend.main:app --reload
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

**Streamlit UI:**
```bash
streamlit run frontend/app.py
# Opens at http://localhost:8501
```

**CLI:**
```bash
# Train a classifier
fft train --data ./images --task classification --num-classes 5

# Run inference
fft predict --image ./test.jpg --weights ./model.pt --decoder linear_probe \
    --task classification --num-classes 5

# List available encoders and decoders
fft info
```

---

## Usage

### Python SDK

The `sdk` module is the recommended interface for notebook and script usage.

#### Classification

```python
import sdk as fft

# Default encoder is dinov3_vitb16 (patch 16, 512×512)
encoder = fft.Encoder("dinov3_vitb16")

head = fft.ClassificationHead(encoder, num_classes=5, head_type="mlp")

dataset = fft.Dataset.from_folder(
    "./my_images/",
    task="classification",
    transform=encoder.get_transform(),
)

trainer = fft.Trainer(head, dataset, lr=1e-3, epochs=20, batch_size=32)
results = trainer.fit()
trainer.save("./classifier.pt")
```

#### Object Detection (RT-DETR, default)

```python
import sdk as fft

encoder = fft.Encoder("dinov2_vitb14")          # RTDETRDecoder sets its intermediate layers
head = fft.DetectionHead(encoder, num_classes=10) # defaults to head_type="rtdetr"

dataset = fft.Dataset.from_folder("./detection_dataset/", task="detection")

trainer = fft.Trainer(
    head, dataset,
    lr=1e-4,
    epochs=50,
    batch_size=8,
    training_mode="deim",     # enables Mosaic augmentation + CDN scheduling
    warmup_epochs=5,
)
results = trainer.fit()
# results keys: best_val_loss, val_map50, val_map, epochs_trained
```

**Detection dataset format** — COCO JSON:
```
detection_dataset/
  images/
    img_001.jpg
  annotations.json       # standard COCO format
```
Boxes must be in `[x, y, w, h]` pixel coordinates in the JSON; the dataset loader normalizes them to `[cx, cy, w, h]` in `[0, 1]`.

#### Semantic Segmentation

```python
import sdk as fft

encoder = fft.Encoder("dinov2_vitb14")
head = fft.SegmentationHead(encoder, num_classes=5, head_type="upernet")
# UPerNet auto-enables intermediate layer extraction

dataset = fft.Dataset.from_folder("./seg_dataset/", task="segmentation")
trainer = fft.Trainer(head, dataset, lr=1e-4, epochs=50)
trainer.fit()
```

**Segmentation dataset format:**
```
seg_dataset/
  images/
    img_001.jpg
  masks/
    img_001.png     # pixel values = class IDs (0-based)
```

#### LoRA Fine-Tuning

Use `lora_rank > 0` in `Trainer` to inject low-rank adapters into the encoder's attention layers. LoRA parameters receive gradients while the rest of the encoder stays frozen.

```python
trainer = fft.Trainer(
    decoder=head,
    train_dataset=dataset,
    lr=1e-4,
    epochs=30,
    lora_rank=4,     # injects LoRA into attn.qkv and attn.proj layers
)
```

LoRA is implemented in `core/encoders/lora.py` as `LoRALinear` (output = W·x + (alpha/rank)·B(A(x))). B is initialized to zeros so training starts from the frozen baseline.

### CLI Reference

```bash
fft train \
    --data ./dataset \
    --task classification|detection|segmentation \
    --num-classes N \
    --encoder dinov3_vitb16 \          # default
    --decoder auto \                    # auto-selects: linear_probe / rtdetr / linear_seg
    --epochs 50 \
    --lr 1e-3 \
    --batch-size 32 \
    --augmentation light \             # none | light | heavy
    --scheduler cosine \               # cosine | step | constant
    --warmup-epochs 5 \
    --early-stopping 10 \
    --output ./output

fft predict \
    --image ./test.jpg \
    --weights ./output/decoder_weights.pt \
    --encoder dinov3_vitb16 \
    --decoder linear_probe \
    --task classification \
    --num-classes 5

fft info      # list all encoders and decoders
```

**Available decoder names** for `--decoder`:

| Task | Names |
|------|-------|
| Classification | `linear_probe`, `mlp`, `transformer` |
| Detection | `rtdetr` (default), `detr_lite`, `detr_multiscale`, `fpn` |
| Segmentation | `linear_seg` (default), `upernet`, `mask_transformer` |

### REST API

```bash
GET  /api/health             # health check
GET  /api/encoders           # list encoder variants
GET  /api/decoders           # list decoder options
GET  /api/projects           # list projects
POST /api/projects           # create project
GET  /api/datasets/{id}      # dataset metadata
POST /api/training/start     # start training job
GET  /api/training/{id}      # job status
WS   /ws/training/{id}       # real-time training progress
```

Full interactive docs are available at `http://localhost:8000/docs` when the backend is running.

---

## Supported Tasks and Decoders

| Task | Decoder | Notes |
|------|---------|-------|
| Classification | `LinearProbe` | Single linear layer on CLS token |
| Classification | `MLPHead` | 2-layer MLP with ReLU and dropout |
| Classification | `TransformerHead` | Cross-attention on patch tokens |
| Detection | `RTDETRDecoder` | **Default.** Multi-scale ViT+CNN, VFL loss, CDN, ~8M params |
| Detection | `DETRLiteDecoder` | Single-scale DETR, simpler, ~3M params |
| Detection | `FPNHead` | Anchor-based FPN, requires intermediate layers |
| Segmentation | `LinearSegHead` | 1×1 conv per patch + upsample |
| Segmentation | `UPerNetHead` | Pyramid pooling, multi-scale |
| Segmentation | `MaskTransformerHead` | Dot-product mask prediction |

---

## Encoders

| Variant | Source | Embed Dim | Patch | Default Input |
|---------|--------|-----------|-------|---------------|
| `dinov3_vits16` | HuggingFace | 384 | 16 | 512×512 |
| `dinov3_vitb16` | HuggingFace | 768 | 16 | 512×512 |
| `dinov3_vitl16` | HuggingFace | 1024 | 16 | 512×512 |
| `dinov2_vits14` | torch.hub | 384 | 14 | 518×518 |
| `dinov2_vitb14` | torch.hub | 768 | 14 | 518×518 |
| `dinov2_vitl14` | torch.hub | 1024 | 14 | 518×518 |
| `dinov2_vitg14` | torch.hub | 1536 | 14 | 518×518 |
| `dinov2_*_reg` | torch.hub | same | 14 | 518×518 |

`dinov3_vitb16` is the system default. DINOv3 includes 4 register tokens (skipped during patch extraction). DINOv2 `_reg` variants have improved attention maps for dense prediction.

Add `intermediate_layers` to extract multi-scale features:

```python
# 4 evenly-spaced layers, required for FPN and UPerNet
encoder = fft.Encoder("dinov2_vitb14")
encoder.model.intermediate_layers = encoder.model.default_intermediate_layers()
```

RTDETRDecoder and FPNHead configure this automatically.

---

## Data Formats

`FFTDataset.from_folder()` auto-detects format based on directory structure.

Direct format loaders are in `core/data/formats.py`:

```python
from core.data.formats import load_coco, load_voc, load_yolo, load_csv

dataset = load_coco("annotations.json", "./images/")
dataset = load_voc("./annotations/", "./images/")
dataset = load_yolo("./labels/", "./images/", class_names=["cat", "dog"])
dataset = load_csv("labels.csv", "./images/")
```

---

## Configuration Reference

All configuration is passed as constructor/function arguments; there are no config files.

### Trainer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | `1e-3` | Learning rate |
| `epochs` | int | `50` | Max training epochs |
| `batch_size` | int | `32` | Batch size |
| `scheduler` | str | `"cosine"` | `"cosine"`, `"step"`, `"constant"` |
| `warmup_epochs` | int | `5` | Linear LR warmup epochs |
| `augmentation` | str | `"light"` | `"none"`, `"light"`, `"heavy"` |
| `early_stopping_patience` | int | `10` | Epochs without improvement before stopping (0 = off) |
| `val_ratio` | float | `0.2` | Train/val split if no val dataset given |
| `num_workers` | int | `4` | DataLoader workers (set to 0 on Windows if issues arise) |
| `accelerator` | str | `"auto"` | PyTorch Lightning accelerator |
| `training_mode` | str | `"standard"` | `"standard"` or `"deim"` (Mosaic + CDN scheduling) |
| `lora_rank` | int | `0` | LoRA rank; 0 disables LoRA |

### RTDETRDecoder Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_queries` | auto | Object queries. Auto: `min(max(C×10, 30), 300)` |
| `num_decoder_layers` | `4` | Transformer decoder depth |
| `hidden_dim` | `256` | Internal feature dimension |
| `num_heads` | `8` | Attention heads |
| `dim_feedforward` | `1024` | FFN intermediate size |
| `max_gt_per_image` | `30` | CDN group size cap |
| `label_noise_ratio` | `0.5` | Label flip probability for CDN positive queries |
| `box_noise_scale` | `1.0` | Box perturbation scale for CDN |

### Environment Variables

None required by default. The following affect behavior:

| Variable | Effect |
|----------|--------|
| `PYTHONNOUSERSITE=1` | Prevents user-site packages from interfering (recommended on dev machine) |
| `TORCH_HOME` | Override torch.hub model cache location |
| `HF_HOME` | Override HuggingFace model cache location |

---

## Testing

```bash
# Run all tests
PYTHONNOUSERSITE=1 pytest tests/

# Run a specific suite
pytest tests/test_decoders/
pytest tests/test_encoders/
pytest tests/test_api/
pytest tests/test_training/

# With coverage
pytest tests/ --cov=core --cov=sdk
```

**Test count:** ~127 tests. Tests use small synthetic tensors and do not download encoder weights; encoder tests mock the model.

**Platform note (Windows):** Set `PYTHONNOUSERSITE=1` to avoid `langsmith` pytest plugin interference. Set `num_workers=0` in any local training scripts to avoid multiprocessing issues.

**FastAPI tests** patch the database engine with an in-memory SQLite instance to avoid touching the filesystem.

---

## Experiments

### Oxford-IIIT Pets Detection

Reproduces a 37-class pet detection experiment using mask-derived bounding boxes.

**Step 1: Prepare data**
```bash
python experiments/prepare_oxford_pets.py
# Downloads dataset, converts segmentation masks to COCO bounding boxes
# Output: experiments/datasets/oxford_pets/  (~3,673 images)
```

**Step 2: Train**
```bash
python experiments/train_deim_pets.py
# dinov2_vits14 + RTDETRDecoder (frozen, no LoRA)
# 60 epochs, batch=8, LR=1e-4, cosine schedule, 20-epoch patience
# Results saved to experiments/results/deim_pets_no_lora/
```

**Known results:** ~14–15% mAP@50 after 20 epochs, still improving at epoch 19. Mosaic augmentation disabled at 50% of training to improve convergence.

---

## Key Concepts

**Frozen encoder**: The ViT backbone weights are never modified during training (unless LoRA is used). `encoder.freeze()` sets `requires_grad=False` on all encoder parameters.

**Feature dict**: `encoder.forward_features(x)` returns a dict with `cls_token`, `patch_tokens`, `spatial_features`, and optionally `intermediate`. Each decoder selects which it needs.

**LoRA (Low-Rank Adaptation)**: Injects trainable rank-r matrices alongside frozen linear layers. Output = W·x + (alpha/rank)·B(A(x)). Only A and B are trained; W stays frozen. Useful when frozen features generalize poorly.

**RTDETRDecoder**: The default detection head. Key improvements over vanilla DETR: multi-scale ViT+CNN features, top-k proposal initialization (warm start), iterative box refinement per decoder layer, varifocal loss (VFL), and contrastive denoising (CDN) during training.

**DEIM mode** (`training_mode="deim"`): Enables Mosaic augmentation on detection datasets (4 images stitched into 1, disabled at epoch `total_epochs * 0.5`) and CDN noise scheduling (CDN disabled in the final 2 epochs).

**Hungarian matching**: Assignment of predicted queries to ground-truth boxes, solved with `scipy.optimize.linear_sum_assignment`. Cost = classification cost + 5×L1 + 2×(−GIoU).

**Varifocal Loss (VFL)**: Classification loss for RT-DETR. Target for matched queries is `IoU(pred_box, gt_box)` rather than a hard 1, giving stronger gradient to high-quality predictions.

**Contrastive Denoising (CDN)**: Training-time technique that prepends noisy GT queries to the decoder. Positive CDN queries start near GT boxes; negative CDN queries start far. The decoder must distinguish them, providing additional supervised signal.

**Intermediate layers**: Multi-scale feature extraction from intermediate ViT blocks. Set via `encoder.intermediate_layers = [i, j, k]` or via `encoder.default_intermediate_layers()`. Required for FPN and UPerNet; set automatically by RTDETRDecoder.

---

## License

Apache-2.0
