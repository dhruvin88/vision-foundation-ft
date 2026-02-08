# Architecture

## Design Philosophy

FFT follows a **frozen encoder + lightweight decoder** pattern. Large pretrained vision foundation models (DINOv2) are loaded and frozen. Only small, task-specific decoder heads are trained, making fine-tuning fast and memory-efficient.

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        Interfaces                            │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │   CLI    │    │  Python SDK  │    │  REST API (FastAPI)│   │
│  └────┬─────┘    └──────┬───────┘    └────────┬──────────┘   │
│       └─────────────────┼─────────────────────┘              │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                    Core Library                       │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │    │
│  │  │ Encoders │ │ Decoders │ │ Training │ │  Data  │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │    │
│  │  ┌────────────┐ ┌────────┐                           │    │
│  │  │ Evaluation │ │ Export │                           │    │
│  │  └────────────┘ └────────┘                           │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Core Modules

### Encoders (`core/encoders/`)

All encoders inherit from `BaseEncoder` and provide:
- `forward(x)`: CLS token output `(B, embed_dim)`
- `forward_features(x)`: Dict with `cls_token`, `patch_tokens`, `spatial_features`
- `freeze()`: Disable gradient computation
- Properties: `embed_dim`, `patch_size`, `num_patches`

**DINOv2Encoder** loads pretrained ViT models from `torch.hub` (facebookresearch/dinov2). All parameters are frozen immediately after loading.

### Decoders (`core/decoders/`)

All decoders inherit from `BaseDecoder` and accept encoder features (not raw images). This separation means:
- The encoder runs once per image (in `torch.no_grad()`)
- Only decoder parameters receive gradients
- Multiple decoders can share the same encoder

Each decoder has a `task` attribute and a `predict(images)` method for end-to-end inference.

**Classification decoders** take the CLS token `(B, embed_dim)` and output logits `(B, num_classes)`.

**Detection decoders** take patch tokens and output bounding box predictions. DETRLite uses learnable object queries with transformer cross-attention. FPNHead builds a feature pyramid for multi-scale detection.

**Segmentation decoders** take spatial features and output per-pixel logits `(B, num_classes, H, W)`.

### Data (`core/data/`)

`FFTDataset` is a unified PyTorch `Dataset` supporting all three tasks. It loads from:
- Folder structure (classification: one folder per class)
- COCO JSON annotations (detection)
- Image/mask pairs (segmentation)

Format converters (`load_coco`, `load_voc`, `load_yolo`, `load_csv`) normalize different annotation formats into the internal sample list format.

Augmentation presets (`none`, `light`, `heavy`) use Albumentations for task-aware transformations.

### Training (`core/training/`)

`Trainer` wraps PyTorch Lightning with sensible defaults:
- Auto train/val split
- Warmup + cosine/step LR scheduling
- Early stopping
- Checkpoint management

`DecoderLightningModule` handles the training loop, loss computation, and metric tracking per task.

### Evaluation (`core/evaluation/`)

- `run_inference()`: Batch inference with auto device selection
- `compute_metrics()`: Task-specific metrics (accuracy, mAP, mIoU)

### Export (`core/export/`)

- `save_decoder_weights()` / `load_decoder_weights()`: Save only decoder parameters with metadata
- `generate_inference_script()`: Auto-generate standalone Python scripts

## Backend (`backend/`)

FastAPI application providing a REST API for project management, dataset upload, training job control, and real-time progress via WebSocket.

**Database**: SQLModel with SQLite (configurable). Models: `Project`, `ImageRecord`, `Annotation`, `TrainingRun`.

**Services**:
- `storage.py`: File management for images, thumbnails, models
- `job_runner.py`: Background training job execution
- `websocket.py`: Real-time training progress updates

## Key Design Decisions

1. **Frozen encoders**: Encoder weights are never modified. This reduces memory usage (no gradients stored for encoder params) and preserves the quality of pretrained features.

2. **Feature dict interface**: Encoders return a dictionary of features (`cls_token`, `patch_tokens`, `spatial_features`). Decoders select which features they need, making the interface flexible without coupling.

3. **PyTorch Lightning**: Training uses Lightning for distributed training support, mixed precision, logging, and checkpointing without custom boilerplate.

4. **Multiple interfaces**: The same core library is accessible via Python SDK (for notebooks/scripts), CLI (for shell workflows), and REST API (for web applications).

5. **Lightweight weight files**: Only decoder weights are saved (typically <50MB), not the large encoder weights (which are loaded from `torch.hub` cache).
