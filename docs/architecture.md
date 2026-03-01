# Architecture

## Design Philosophy

FFT follows a **frozen encoder + lightweight decoder** pattern. Large pretrained vision foundation models (DINOv3, DINOv2) are loaded and frozen. Only small, task-specific decoder heads are trained, making fine-tuning fast and memory-efficient.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Interfaces                                 │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  ┌───────────┐  │
│  │   CLI    │  │  Python SDK  │  │ REST API      │  │ Streamlit │  │
│  │ fft ...  │  │  import sdk  │  │ FastAPI/uvicorn│  │ frontend/ │  │
│  └────┬─────┘  └──────┬───────┘  └───────┬───────┘  └─────┬─────┘  │
│       └───────────────┼─────────────────┬┘                │        │
│                       ▼                 ▼                  │        │
│  ┌────────────────────────────────────────────────────┐   │        │
│  │                    Core Library                     │ ◄─┘        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │            │
│  │  │ Encoders │ │ Decoders │ │ Training │ │  Data  │ │            │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘ │            │
│  │  ┌────────────┐ ┌────────┐                          │            │
│  │  │ Evaluation │ │ Export │                          │            │
│  │  └────────────┘ └────────┘                          │            │
│  └────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
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

#### LoRA (`core/encoders/lora.py`)

When frozen features are insufficient, LoRA adapters can be injected into the encoder's attention layers:

```python
encoder.enable_lora(rank=4, alpha=4.0, target_modules=["attn.qkv", "attn.proj"])
```

`apply_lora()` replaces each target `nn.Linear` with a `LoRALinear` wrapper:

```
output = W·x + (alpha/rank) · B(A(x))
```

- `W` stays frozen; only `A` (rank×in) and `B` (out×rank) are trained.
- `B` is initialized to zeros, so LoRA starts as an identity transformation.
- The encoder's `_fwd_ctx` property returns `contextlib.nullcontext()` when LoRA is active (so gradients flow) and `torch.no_grad()` otherwise.
- `DecoderLightningModule` re-enables LoRA parameters after calling `encoder.freeze()` to ensure they keep `requires_grad=True`.

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

**RTDETRDecoder** (default, multi-scale):

See [rtdetr.md](rtdetr.md) for a full architecture walkthrough. Summary:
- Fuses 3-scale ViT intermediate features with a parallel CNN branch (SpatialPriorModule)
- Applies a HybridEncoder (transformer on coarsest scale + FPN top-down)
- Top-k proposal initialization warm-starts object queries from high-scoring encoder positions
- 4 decoder layers with iterative box refinement
- Varifocal loss (VFL) + L1 + GIoU for classification and localization
- Contrastive denoising (CDN) during training for additional supervised signal

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

**Training modes:**
- `"standard"` (default): Standard training loop.
- `"deim"`: Detection-focused. Wraps the dataset with `MosaicDetectionDataset` (4-image grid stitching, disabled at `epoch >= total_epochs * 0.5`) and disables CDN in the final 2 epochs.

`DecoderLightningModule` handles the training loop, loss computation, and metric tracking per task.

**Loss routing** (detection):
- If the prediction dict contains `"enc_outputs"` (RTDETRDecoder output), the full RT-DETR loss is used: VFL + L1 + GIoU on final layer, same losses on all 4 aux layers (reusing same Hungarian matches), encoder auxiliary BCE, and CDN loss if CDN outputs are present.
- Otherwise (DETRLiteDecoder), the DETR loss is used: Hungarian matching + CrossEntropy (with `eos_coef=0.1` for no-object) + L1 + GIoU.

**Validation metrics:**
- Classification: `val_acc` (logged per epoch)
- Detection: `val_map50` and `val_map` via `torchmetrics.detection.MeanAveragePrecision`
- Segmentation: `val_loss` (cross-entropy)

### Evaluation (`core/evaluation/`)

- `run_inference()`: Batch inference with auto device selection
- `compute_metrics()`: Task-specific metrics (accuracy, mAP, mIoU)

### Export (`core/export/`)

- `save_decoder_weights()` / `load_decoder_weights()`: Save only decoder parameters with metadata
- `generate_inference_script()`: Auto-generate standalone Python scripts

## Backend (`backend/`)

FastAPI application providing a REST API for project management, dataset upload, training job control, and real-time progress via WebSocket.

**Database**: SQLModel with SQLite (configurable). Models: `Project`, `ImageRecord`, `Annotation`, `TrainingRun`. Database is created at `./data/app.db` on startup via `on_startup`.

**Routes** (`backend/api/`):
- `projects.py`: CRUD for training projects
- `datasets.py`: Dataset upload and management
- `training.py`: Start/stop/monitor training jobs
- `models.py`: Model listing and download

**Services** (`backend/services/`):
- `storage.py`: File management for images, thumbnails, models
- `job_runner.py`: Background training job execution
- `websocket.py`: Real-time training progress updates via WebSocket

**Note:** The `/api/decoders` endpoint does not yet list `rtdetr` — it reflects an older decoder set. The CLI and SDK are the authoritative sources for available decoder names.

## Frontend (`frontend/`)

Streamlit application with a 5-page pipeline:

```
app.py              # Entry point, sidebar navigation, task selection
pages/
  1_Dataset.py     # Dataset path/upload configuration
  2_Model.py       # Encoder and decoder selection
  3_Training.py    # Training parameter configuration and job launch
  4_Results.py     # Loss curves and metric display
  5_Inference.py   # Run predictions on uploaded images
```

Start with: `streamlit run frontend/app.py`

## Key Design Decisions

1. **Frozen encoders**: Encoder weights are never modified by default. This reduces memory usage (no gradients stored for encoder params) and preserves the quality of pretrained features.

2. **LoRA as an opt-in escape hatch**: When frozen features are insufficient, `lora_rank > 0` injects trainable low-rank adapters into attention layers. The encoder stays structurally frozen; only the LoRA parameters (A, B matrices) are trained. This avoids full fine-tuning while still allowing adaptation.

3. **Feature dict interface**: Encoders return a dictionary of features (`cls_token`, `patch_tokens`, `spatial_features`, optional `intermediate`). Decoders select which keys they need, making the interface flexible without coupling encoder to decoder.

4. **RTDETRDecoder as the default detection head**: RT-DETR converges in 20–30 epochs vs. 50–100 for vanilla DETR, due to warm-start query initialization and better loss (VFL over cross-entropy). This is the default for `DetectionHead()` and `fft train --task detection`.

5. **PyTorch Lightning**: Training uses Lightning for distributed training support, mixed precision, logging, and checkpointing without custom boilerplate.

6. **Multiple interfaces**: The same core library is accessible via Python SDK (for notebooks/scripts), CLI (for shell workflows), REST API (for web applications), and Streamlit UI (for interactive browser-based use).

7. **Lightweight weight files**: Only decoder weights are saved (typically <50 MB), not the large encoder weights (which are loaded from HuggingFace/torch.hub cache on demand).

## Complete Training Flow

```
┌──────────────────┐
│  Input Image     │
│   (B, 3, H, W)   │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Encoder (DINOv3/DINOv2)                    │
│  ┌────────────────────────────────────────┐ │
│  │  Patch Embedding → Transformer Blocks  │ │
│  │  → Layer Norm                          │ │
│  │                                        │ │
│  │  [Optional] LoRA adapters on attn      │ │
│  │  layers: output += (α/r)·B(A(x))       │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  Frozen params: no gradients                │
│  LoRA params (if enabled): gradients flow   │
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
│  • RT-DETR: VFL + L1 + GIoU + CDN + aux    │
│  • DETRLite: Hungarian + CE + L1 + GIoU     │
│  • CrossEntropy (segmentation)              │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
         Backprop through decoder only
         (+ LoRA params if enabled)
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Optimizer Step (AdamW)                     │
│  • LR scheduler (warmup + cosine/step)      │
│  • Updates decoder params + LoRA (if on)    │
└─────────────────────────────────────────────┘
```

**Memory efficiency**: Since encoder gradients are never computed (in standard mode), GPU memory usage is dominated by decoder activations and optimizer states. A 1B-parameter encoder + 10M-parameter decoder uses similar memory to training a 10M-parameter model from scratch. With LoRA, a small additional overhead is incurred for the LoRA A/B parameters and their gradients.
