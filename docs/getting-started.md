# Getting Started

## Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended; CPU training works but is slower)

## Installation

```bash
git clone https://github.com/your-org/foundation-model-any-ft.git
cd foundation-model-any-ft

# Install core library
pip install -e .

# Install with backend support
pip install -e ".[backend]"

# Install with development tools
pip install -e ".[dev]"

# Install everything
pip install -e ".[all]"
```

Encoder weights download automatically on first use:
- DINOv3 models download from HuggingFace (`facebook/dinov3-*`)
- DINOv2 models download from `torch.hub` (`facebookresearch/dinov2`)

---

## Your First Classification Model

### 1. Prepare Your Data

Organize images into folders by class:

```
my_dataset/
  cats/
    img_001.jpg
    img_002.jpg
  dogs/
    img_001.jpg
    img_002.jpg
```

### 2. Train with the SDK

```python
import sdk as fft

# Default encoder: dinov3_vitb16 (768-dim, patch 16, 512×512)
encoder = fft.Encoder("dinov3_vitb16")

# Create a classification head
head = fft.ClassificationHead(encoder, num_classes=2, head_type="linear")

# Load dataset with encoder's transform pipeline
dataset = fft.Dataset.from_folder(
    "./my_dataset/",
    task="classification",
    transform=encoder.get_transform(),
)

# Train (auto-splits into 80% train / 20% val)
trainer = fft.Trainer(head, dataset, lr=1e-3, epochs=20, batch_size=32)
results = trainer.fit()
print(results)
# {"best_val_loss": ..., "val_acc": ..., "epochs_trained": ..., "best_model_path": ...}

# Save decoder weights (encoder not saved — reload from hub on inference)
trainer.save("./classifier.pt")
```

### 3. Or Train with the CLI

```bash
fft train \
    --data ./my_dataset \
    --task classification \
    --num-classes 2 \
    --encoder dinov3_vitb16 \
    --decoder linear_probe \
    --epochs 20 \
    --output ./output
```

### 4. Run Inference

```python
import sdk as fft
from core.export.weights import load_decoder_weights

encoder = fft.Encoder("dinov3_vitb16")
head = fft.ClassificationHead(encoder, num_classes=2)
load_decoder_weights(head, "./classifier.pt")

results = fft.run_inference(head, ["./test_image.jpg"])
print(results)
```

---

## Object Detection

### Data Format

Expects COCO JSON:

```
detection_dataset/
  images/
    img_001.jpg
    img_002.jpg
  annotations.json    # standard COCO format
```

Boxes in the JSON should be `[x, y, w, h]` in pixel coordinates. The dataset loader converts them to normalized `[cx, cy, w, h]` in `[0, 1]` automatically.

### Train with RTDETRDecoder (default)

```python
import sdk as fft

encoder = fft.Encoder("dinov2_vitb14")
# head_type="rtdetr" is the default — no need to specify it
head = fft.DetectionHead(encoder, num_classes=10)

dataset = fft.Dataset.from_folder("./detection_dataset/", task="detection")

trainer = fft.Trainer(
    head, dataset,
    lr=1e-4,
    epochs=50,
    batch_size=8,
    warmup_epochs=5,
    training_mode="deim",   # enables Mosaic + CDN scheduling
)
results = trainer.fit()
# results["val_map50"] and results["val_map"] are available for detection
```

**Note:** Detection typically requires 20–50 epochs to converge. mAP@50 of 0% in the first few epochs is normal — the decoder is warming up.

### DEIM Mode

`training_mode="deim"` adds two training strategies for detection:
- **Mosaic augmentation**: 4 images stitched into one. Disabled automatically after `total_epochs * 0.5` epochs to allow clean convergence.
- **CDN scheduling**: Contrastive denoising disabled in the final 2 epochs.

---

## Semantic Segmentation

### Data Format

```
seg_dataset/
  images/
    img_001.jpg
    img_002.jpg
  masks/
    img_001.png   # Pixel values = class IDs (0-based integers)
    img_002.png
```

### Train

```python
import sdk as fft

encoder = fft.Encoder("dinov2_vitb14")
head = fft.SegmentationHead(encoder, num_classes=5, head_type="upernet")
# UPerNet auto-enables intermediate layer extraction

dataset = fft.Dataset.from_folder("./seg_dataset/", task="segmentation")
trainer = fft.Trainer(head, dataset, lr=1e-4, epochs=50)
trainer.fit()
```

---

## LoRA Fine-Tuning

When frozen encoder features do not generalize to your domain, enable LoRA to partially fine-tune the encoder's attention layers:

```python
import sdk as fft

encoder = fft.Encoder("dinov2_vitb14")
head = fft.ClassificationHead(encoder, num_classes=10, head_type="mlp")
dataset = fft.Dataset.from_folder("./my_data/", task="classification")

trainer = fft.Trainer(
    head, dataset,
    lr=1e-4,
    epochs=30,
    lora_rank=4,    # injects LoRA into attn.qkv and attn.proj (default targets)
)
trainer.fit()
```

LoRA adds trainable `A` and `B` matrices of rank `r` alongside each target linear layer. The scaling is `alpha/rank` (default `alpha=4.0, rank=4`). `B` is initialized to zeros so training starts from the pretrained baseline.

---

## Choosing an Encoder

| Variant | Source | Embed Dim | Input Size | Speed |
|---------|--------|-----------|------------|-------|
| `dinov3_vits16` | HuggingFace | 384 | 512×512 | Fastest |
| `dinov3_vitb16` | HuggingFace | 768 | 512×512 | Fast (default) |
| `dinov3_vitl16` | HuggingFace | 1024 | 512×512 | Moderate |
| `dinov2_vits14` | torch.hub | 384 | 518×518 | Fastest |
| `dinov2_vitb14` | torch.hub | 768 | 518×518 | Fast |
| `dinov2_vitl14` | torch.hub | 1024 | 518×518 | Moderate |
| `dinov2_vitg14` | torch.hub | 1536 | 518×518 | Slowest |

Add `_reg` to any DINOv2 variant (e.g., `dinov2_vitb14_reg`) for register token variants with improved attention maps — beneficial for detection and segmentation.

---

## Choosing a Decoder

### Classification
- **linear** (`LinearProbe`): Single linear layer, strong baseline, fastest to train.
- **mlp** (`MLPHead`): 2-layer MLP with ReLU and dropout. Slightly more capacity.
- **transformer** (`TransformerHead`): Cross-attention from learnable class queries to patch tokens. Most expressive.

### Detection
- **rtdetr** (`RTDETRDecoder`) — default: RT-DETRv2-inspired. Multi-scale ViT+CNN fusion, VFL loss, CDN. Best mAP, ~8M params, converges in 20–30 epochs.
- **detr_lite** (`DETRLiteDecoder`): Single-scale DETR. Simpler, ~3M params, needs 50–100 epochs.
- **fpn** (`FPNHead`): Anchor-based Feature Pyramid Network. Requires intermediate layers (auto-set).

### Segmentation
- **linear** (`LinearSegHead`): Per-patch 1×1 conv + bilinear upsample. Fastest.
- **upernet** (`UPerNetHead`): Pyramid pooling + top-down FPN. Strongest for dense scenes.
- **mask_transformer** (`MaskTransformerHead`): Learnable class queries dot-producted with patch tokens.

---

## Next Steps

- [SDK Reference](sdk-reference.md) — full API documentation
- [Architecture](architecture.md) — component deep-dives and data flow diagrams
- [RTDETRDecoder Guide](rtdetr.md) — detailed architecture of the default detection head
