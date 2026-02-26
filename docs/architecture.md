# Architecture

## Design Philosophy

FFT follows a **frozen encoder + lightweight decoder** pattern. Large pretrained vision foundation models (DINOv3, DINOv2) are loaded and frozen. Only small, task-specific decoder heads are trained, making fine-tuning fast and memory-efficient.

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
- `forward_features(x)`: Dict with `cls_token`, `patch_tokens`, `spatial_features`, optionally `intermediate`
- `freeze()`: Disable gradient computation
- Properties: `embed_dim`, `patch_size`, `num_patches`, `intermediate_layers`

**Available encoders:**
- **DINOv3Encoder** (default): Loads from HuggingFace (`facebook/dinov3-*`). Variants: vits16, vitb16, vitl16. Patch size 16, default input 512×512.
- **DINOv2Encoder**: Loads from `torch.hub` (`facebookresearch/dinov2`). Variants: vits14, vitb14, vitl14, vitg14 (+ `_reg`). Patch size 14, default input 518×518.

All parameters are frozen immediately after loading. Use `create_encoder(model_name)` factory to instantiate.

#### Encoder Feature Flow

```
Input Image (B, 3, H, W)
        ↓
┌───────────────────────────────────────────┐
│   Vision Transformer (Frozen)              │
│  ┌─────────────────────────────────────┐   │
│  │  Patch Embedding (conv proj)        │   │
│  │    → (B, N, embed_dim)              │   │
│  └─────────────────────────────────────┘   │
│               ↓                             │
│  ┌─────────────────────────────────────┐   │
│  │  Transformer Blocks × L             │   │
│  │   (self-attention + FFN)            │   │
│  │                                     │   │
│  │   [CLS, reg0-3, patch1..patchN]    │ ← DINOv3 includes 4 register tokens
│  └─────────────────────────────────────┘   │
│               ↓                             │
│  ┌─────────────────────────────────────┐   │
│  │  Layer Norm                         │   │
│  └─────────────────────────────────────┘   │
└───────────────────────────────────────────┘
        ↓
forward_features() extracts:
┌──────────────────────────────────────────┐
│  cls_token:         (B, D)               │ ← Global image representation
│  patch_tokens:      (B, N, D)            │ ← Per-patch features (flattened)
│  spatial_features:  (B, D, h, w)         │ ← Reshaped to spatial grid
│  intermediate:      [(B, D, h, w), ...]  │ ← Optional multi-scale (if enabled)
└──────────────────────────────────────────┘

where:
  B = batch size
  D = embed_dim (384/768/1024/1536)
  N = num_patches = (input_size/patch_size)²
  h = w = grid_size = input_size/patch_size
  L = num_blocks (12 for S/B, 24 for L, 40 for G)
```

**Multi-scale features**: Set `encoder.intermediate_layers = [2, 5, 8, 11]` to extract features from intermediate transformer blocks. This is required for FPN (detection) and UPerNet (segmentation) decoders.

### Decoders (`core/decoders/`)

All decoders inherit from `BaseDecoder` and accept encoder features (not raw images). This separation means:
- The encoder runs once per image (in `torch.no_grad()`)
- Only decoder parameters receive gradients
- Multiple decoders can share the same encoder

Each decoder has a `task` attribute and a `predict(images)` method for end-to-end inference.

#### Classification Decoders

Take the **CLS token** `(B, embed_dim)` and output logits `(B, num_classes)`.

```
Encoder features['cls_token']  (B, D)
        ↓
┌─────────────────────────────────┐
│  LinearProbe                    │  ← Single linear layer (simplest)
│    Linear(D → num_classes)      │
└─────────────────────────────────┘
        OR
┌─────────────────────────────────┐
│  MLPHead                        │  ← 2-layer MLP with dropout
│    Linear(D → hidden)           │
│    → ReLU → Dropout             │
│    → Linear(hidden → classes)   │
└─────────────────────────────────┘
        OR
┌─────────────────────────────────┐
│  TransformerHead                │  ← Cross-attention decoder
│    Learnable class queries      │
│    → Cross-attend to patches    │
│    → MLP → logits               │
└─────────────────────────────────┘
        ↓
    (B, num_classes)
```

#### Detection Decoders

Take **spatial/patch features** and output bounding boxes + class logits.

**DETRLite** (single-scale, transformer-based):
```
Encoder features['spatial_features']  (B, D, h, w)
        ↓
┌────────────────────────────────────────┐
│  Input Projection                      │
│    Conv(D → hidden_dim)                │
└────────────────────────────────────────┘
        ↓  (B, hidden_dim, h, w)
┌────────────────────────────────────────┐
│  Learnable Object Queries              │
│    (num_queries, hidden_dim)           │  ← e.g., 100 queries
│                                        │
│  Transformer Decoder Layers            │
│    Self-Attention(queries)             │
│    → Cross-Attention(queries, features)│
│    → FFN                               │
│    (repeat × num_decoder_layers)       │
└────────────────────────────────────────┘
        ↓  (B, num_queries, hidden_dim)
┌────────────────────────────────────────┐
│  Detection Heads                       │
│    Class head:  Linear → (B, Q, C+1)   │  ← C classes + no-object
│    Box head:    MLP → (B, Q, 4)        │  ← [cx, cy, w, h] normalized
└────────────────────────────────────────┘
```

**FPNHead** (multi-scale, anchor-based):
```
Encoder features['intermediate']  [(B,D,h₁,w₁), (B,D,h₂,w₂), ...]
        ↓                          4 scales from layers [2,5,8,11]
┌────────────────────────────────────────┐
│  Feature Pyramid Network               │
│                                        │
│  Lateral convs: each scale → channels │
│  Top-down path: upsample + add         │
│                                        │
│    P4 ← Conv(layer11) ──────────┐      │
│    P3 ← Conv(layer8)  + ↑P4 ────┤      │
│    P2 ← Conv(layer5)  + ↑P3 ────┤      │
│    P1 ← Conv(layer2)  + ↑P2 ────┘      │
└────────────────────────────────────────┘
        ↓  [P1, P2, P3, P4]  multi-scale pyramid
┌────────────────────────────────────────┐
│  Detection Heads (per pyramid level)   │
│    Conv → class logits                 │
│    Conv → bbox deltas                  │
│  + Anchor generation per scale         │
└────────────────────────────────────────┘
```

#### Segmentation Decoders

Take **spatial features** and output per-pixel logits `(B, num_classes, H, W)`.

**LinearSegHead** (single-scale, per-patch classification):
```
Encoder features['spatial_features']  (B, D, h, w)
        ↓
┌────────────────────────────────────────┐
│  Conv(D → num_classes, kernel=1)       │  ← 1×1 conv (per-patch classifier)
└────────────────────────────────────────┘
        ↓  (B, num_classes, h, w)
┌────────────────────────────────────────┐
│  Bilinear Upsample                     │  ← Restore to input resolution
│    (h, w) → (H, W)                     │
└────────────────────────────────────────┘
        ↓
    (B, num_classes, H, W)
```

**UPerNetHead** (multi-scale, pyramid pooling):
```
Encoder features['intermediate']  [(B,D,h₁,w₁), (B,D,h₂,w₂), ...]
        ↓
┌────────────────────────────────────────┐
│  Pyramid Pooling Module (PPM)          │
│    Global pool → 1×1                   │
│    AdaptivePool → 2×2, 4×4, 8×8        │
│    All upsampled + concat              │
└────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────┐
│  FPN-like Decoder                      │
│    Top-down fusion of pyramid levels   │
│    Conv refinement per level           │
└────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────┐
│  Segmentation Head                     │
│    Conv → upsample → logits            │
└────────────────────────────────────────┘
        ↓
    (B, num_classes, H, W)
```

**MaskTransformerHead** (transformer-based, patch-wise):
```
Encoder features['patch_tokens']  (B, N, D)
        ↓
┌────────────────────────────────────────┐
│  Learnable Class Queries               │
│    (num_classes, D)                    │
│                                        │
│  Transformer Decoder                   │
│    Self-attention on queries           │
│    → Cross-attention to patch tokens   │
│    → FFN                               │
└────────────────────────────────────────┘
        ↓  (B, num_classes, D)
┌────────────────────────────────────────┐
│  Mask Generation                       │
│    Dot-product: queries @ patch_tokens │
│    → (B, num_classes, N)               │
│    → Reshape to (B, num_classes, h, w) │
│    → Upsample to (B, num_classes, H, W)│
└────────────────────────────────────────┘
```

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

5. **Lightweight weight files**: Only decoder weights are saved (typically <50MB), not the large encoder weights (which are loaded from HuggingFace/torch.hub cache).

## Complete Training Flow

```
┌──────────────────┐
│  Input Image     │
│   (B, 3, H, W)   │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Frozen Encoder (DINOv3/DINOv2)             │
│  ┌────────────────────────────────────────┐ │
│  │  Patch Embedding → Transformer Blocks  │ │
│  │  → Layer Norm                          │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  No gradients computed (frozen params)      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
         features dict {
           'cls_token': (B, D),
           'patch_tokens': (B, N, D),
           'spatial_features': (B, D, h, w),
           'intermediate': [...] (optional)
         }
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Trainable Decoder                          │
│  ┌────────────────────────────────────────┐ │
│  │  Task-specific head (MLP/DETR/UPerNet) │ │
│  │  Parameters: ~1M-50M (vs 20M-1.5B enc) │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  Gradients flow only through decoder        │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
         Task-specific output
         • Classification: (B, num_classes)
         • Detection: boxes + labels
         • Segmentation: (B, num_classes, H, W)
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Loss Computation                           │
│  • CrossEntropy (classification)            │
│  • Hungarian + L1 + GIoU (DETR detection)   │
│  • Focal Loss (FPN detection)               │
│  • CrossEntropy (segmentation)              │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
         Backprop through decoder only
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Optimizer Step (AdamW)                     │
│  • LR scheduler (warmup + cosine/step)      │
│  • Updates only decoder params              │
└─────────────────────────────────────────────┘
```

**Memory efficiency**: Since encoder gradients are never computed, GPU memory usage is dominated by decoder activations and optimizer states. A 1B-parameter encoder + 10M-parameter decoder uses similar memory to training a 10M-parameter model from scratch.
