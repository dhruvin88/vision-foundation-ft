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

Core dependencies include `transformers>=4.50.0` and `bitsandbytes>=0.41.0`. Encoder weights download automatically on first use:
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

Masks must be grayscale PNGs where each pixel value is a class ID. For Oxford Pets trimap segmentation the mapping is: 0=background, 1=pet, 2=boundary.

### Prepare Oxford Pets Segmentation Data

```bash
python experiments/prepare_oxford_pets_seg.py
# Downloads Oxford Pets trimaps (~800 MB), remaps to 3-class masks
# Output: experiments/datasets/oxford_pets_seg/
#   images/  — JPEG images
#   masks/   — grayscale PNGs with values 0/1/2
```

### Train Locally (Smoke Test)

```bash
python experiments/train_seg_pets.py
# dinov2_vits14 + UPerNetHead, 3 classes, input_size=224
# 5 epochs, batch=4, for quick validation
```

### Train with the SDK

```python
import sdk as fft

encoder = fft.Encoder("dinov2_vitb14")
head = fft.SegmentationHead(encoder, num_classes=3, head_type="upernet")
# UPerNet auto-enables intermediate layer extraction

# output_size resizes masks to match encoder input size using Image.NEAREST
dataset = fft.Dataset.from_folder(
    "./seg_dataset/",
    task="segmentation",
    transform=encoder.get_transform(),
    output_size=518,   # set to encoder input size (518 for dinov2_vitb14 default)
)
trainer = fft.Trainer(head, dataset, lr=1e-4, epochs=30)
trainer.fit()
```

**Why `output_size`?** The raw mask PNG may have a different resolution than the encoder's expected input. When `output_size` is set, `FFTDataset` resizes each mask to `output_size × output_size` using `Image.NEAREST` (preserving integer class IDs, no interpolation artifacts). The same value should match the encoder input size so that the spatial grid dimensions align.

### Train on Modal A10G

For full-resolution, longer training runs:

```bash
# One-time dataset upload to Modal volume
modal run experiments/modal_train_seg.py::upload_dataset

# Run training (30 epochs, LoRA rank=4, input_size=448, A10G)
modal run --detach experiments/modal_train_seg.py

# Download results
modal volume get fft-results /seg_pets ./experiments/results_modal/seg_pets
```

Configurable via CLI arguments:
```bash
# Larger encoder, quick smoke test:
modal run experiments/modal_train_seg.py -- --encoder dinov2_vitb14 --epochs 3
```

**Result (Oxford Pets):** val_loss=0.1375 (15 epochs, early stopped on A10G).

---

## Visual Question Answering (VLM)

The VLM decoder uses a LLaVA 1.5-style architecture: DINOv2 patch tokens → MLP projector → Phi-3.5-mini-instruct. Training is two-stage.

### Prerequisites

```bash
pip install transformers accelerate sentencepiece
python experiments/prepare_oxford_pets.py   # if not already done
```

### Train Locally (TinyLlama, CPU/GPU)

```bash
python experiments/train_vlm_pets.py
# dinov2_vits14 + TinyLlama-1.1B (local default)
# Stage 1: 3 epochs projector alignment (lr=1e-3)
# Stage 2: 10 epochs instruction tuning (lr=2e-5, LoRA rank=8)
# Results: experiments/results/vlm_pets/
```

### Train on Modal A10G (Phi-3.5-mini)

```bash
# One-time dataset upload (reuses fft-datasets volume if detection already uploaded)
modal run experiments/modal_train_vlm.py::upload_dataset

# Run training (Stage 1: 3 epochs + Stage 2: 10 epochs)
modal run experiments/modal_train_vlm.py

# Download results
modal volume get fft-results /vlm_pets ./experiments/results_modal/vlm_pets
```

Configurable arguments:
```bash
# Larger encoder:
modal run experiments/modal_train_vlm.py -- --encoder dinov2_vitb14

# Quick smoke-test (1+2 epochs):
modal run experiments/modal_train_vlm.py -- --s1-epochs 1 --s2-epochs 2
```

**Result (Oxford Pets):** val_token_acc=99.15% (13 epochs, A10G).

### Using the VLM API Directly

```python
from core.encoders import create_encoder
from core.decoders.vlm import VLMDecoder
from core.data.vqa_dataset import PetsVQADataset
from core.training.vlm_trainer import VLMTrainer

encoder = create_encoder("dinov2_vits14", input_size=224)

# Stage 1: projector alignment only
decoder = VLMDecoder(
    encoder,
    llm_name="microsoft/Phi-3.5-mini-instruct",
    freeze_llm=True,
    lora_rank=0,
    pool_patches=2,    # 16×16 → 8×8 = 64 visual tokens
)

dataset = PetsVQADataset(
    annotations_json="./datasets/oxford_pets/annotations.json",
    images_dir="./datasets/oxford_pets/images/",
    tokenizer=decoder.tokenizer,
    transform=encoder.get_transform(),
)
train_ds, val_ds = dataset.split()

stage1 = VLMTrainer(
    decoder, train_ds, val_ds,
    lr=1e-3, epochs=3, batch_size=4,
    stage=1, warmup_epochs=1,
)
results1 = stage1.fit()

# Stage 2: unlock LLM LoRA, retrain
decoder.enable_llm_lora(rank=8)
stage2 = VLMTrainer(
    decoder, train_ds, val_ds,
    lr=2e-5, epochs=10, batch_size=4,
    stage=2, warmup_epochs=2, early_stopping_patience=5,
)
results2 = stage2.fit()
print(f"val_token_acc: {results2['val_token_acc']:.4f}")
```

### 4-Bit Inference on Machines with Limited VRAM

```python
from core.encoders import create_encoder
from core.decoders.vlm import VLMDecoder

encoder = create_encoder("dinov2_vits14", input_size=224)
encoder.to("cuda")

decoder = VLMDecoder(
    encoder,
    llm_name="microsoft/Phi-3.5-mini-instruct",
    load_in_4bit=True,   # NF4 quantization; LLM placed on cuda:0 automatically
)
# Only move the projector manually — do NOT call decoder.to("cuda")
decoder.projector.to("cuda")

# Run generation
import torch
image_t = encoder.get_transform()(image).unsqueeze(0).to("cuda")
with torch.no_grad():
    features = encoder.forward_features(image_t)

enc = decoder.tokenizer(
    "<|user|>\nWhat breed is this pet?<|end|>\n<|assistant|>\n",
    return_tensors="pt",
)
answers = decoder.generate(
    features,
    enc["input_ids"].to("cuda"),
    enc["attention_mask"].to("cuda"),
    max_new_tokens=32,
)
print(answers[0])
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

DINOv2 is preferred over DINOv3 for the VLM and segmentation tasks in the benchmarks above (`dinov2_vits14`, `input_size=224` for VLM; `dinov2_vits14`, `input_size=448` for segmentation).

---

## Choosing a Decoder

### Classification
- **linear** (`LinearProbe`): Single linear layer, strong baseline, fastest to train.
- **mlp** (`MLPHead`): 2-layer MLP with ReLU and dropout. Slightly more capacity.
- **transformer** (`TransformerHead`): Cross-attention from learnable class queries to patch tokens. Most expressive.

### Detection
- **rtdetr** (`RTDETRDecoder`) — default: RT-DETRv2-inspired. Multi-scale ViT+CNN fusion, VFL loss, CDN. Best mAP, ~8M params, converges in 20–50 epochs.
- **detr_lite** (`DETRLiteDecoder`): Single-scale DETR. Simpler, ~3M params, needs 50–100 epochs.
- **fpn** (`FPNHead`): Anchor-based Feature Pyramid Network. Requires intermediate layers (auto-set).

### Segmentation
- **linear** (`LinearSegHead`): Per-patch 1×1 conv + bilinear upsample. Fastest.
- **upernet** (`UPerNetHead`): Pyramid pooling + top-down FPN. Strongest for dense scenes.
- **mask_transformer** (`MaskTransformerHead`): Learnable class queries dot-producted with patch tokens.

### VLM
- `VLMDecoder` — MLP projector + causal LLM (Phi-3.5-mini-instruct or TinyLlama). Use the core API directly; not available in the `sdk` module.

---

## Next Steps

- [SDK Reference](sdk-reference.md) — full API documentation
- [Architecture](architecture.md) — component deep-dives and data flow diagrams
- [RTDETRDecoder Guide](rtdetr.md) — detailed architecture of the default detection head
