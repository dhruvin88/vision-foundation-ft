# SDK Reference

The `sdk` module provides a high-level API for fine-tuning vision foundation models for classification, detection, and segmentation. The VLM decoder is accessed via the core library directly — see [VLMDecoder](#vlmdecoder) below.

```python
import sdk as fft
```

---

## Encoder

```python
encoder = fft.Encoder(
    model_name="dinov3_vitb16",     # default
    input_size=None,                # uses encoder's default if None
    intermediate_layers=None,       # list of layer indices for multi-scale features
)
```

**Parameters:**
- `model_name` (str): Encoder variant. DINOv3: `dinov3_vits16`, `dinov3_vitb16`, `dinov3_vitl16`. DINOv2: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14` (and `_reg` variants). Default: `"dinov3_vitb16"`.
- `input_size` (int | None): Input image size. Must be divisible by patch size (16 for DINOv3, 14 for DINOv2). Uses encoder default if `None`.
- `intermediate_layers` (list[int] | None): Layer indices for multi-scale feature extraction. Required for FPN and UPerNet; RTDETRDecoder sets this automatically.

**Properties:**
- `embed_dim` (int): Output embedding dimension.
- `model`: The underlying `DINOv2Encoder` or `DINOv3Encoder` instance.

**Methods:**
- `get_transform()`: Returns a `torchvision.transforms.Compose` pipeline for this encoder (resize + normalize).

---

## ClassificationHead

```python
head = fft.ClassificationHead(encoder, num_classes=10, head_type="linear")
```

**Parameters:**
- `encoder`: An `Encoder` instance or `DINOv2Encoder` / `DINOv3Encoder` directly.
- `num_classes` (int): Number of output classes.
- `head_type` (str): One of `"linear"`, `"mlp"`, `"transformer"`. Default: `"linear"`.

**Head types:**

| Type | Class | Input | Description |
|------|-------|-------|-------------|
| `"linear"` | `LinearProbe` | CLS token | Single linear layer |
| `"mlp"` | `MLPHead` | CLS token | 2-layer MLP with ReLU and dropout |
| `"transformer"` | `TransformerHead` | Patch tokens | Cross-attention decoder with learnable queries |

---

## DetectionHead

```python
head = fft.DetectionHead(encoder, num_classes=10, head_type="rtdetr", **kwargs)
```

**Parameters:**
- `encoder`: An `Encoder` instance or encoder directly.
- `num_classes` (int): Number of object classes (no background class needed for RT-DETR).
- `head_type` (str): One of `"rtdetr"`, `"detr_lite"`, `"fpn"`. Default: `"rtdetr"`.
- `**kwargs`: Passed to the decoder constructor.

**Head types:**

| Type | Class | Description |
|------|-------|-------------|
| `"rtdetr"` | `RTDETRDecoder` | Multi-scale ViT+CNN, focal BCE loss, CDN. Default. |
| `"detr_lite"` | `DETRLiteDecoder` | Single-scale DETR with learnable queries |
| `"fpn"` | `FPNHead` | Feature Pyramid Network with anchor-based detection |

**RTDETRDecoder output:** `{"pred_logits": (B, Q, C), "pred_boxes": (B, Q, 4), "aux_outputs": [...], "enc_outputs": {...}}`

- `pred_logits`: Raw logits. Apply `.sigmoid()` for class scores (no background class).
- `pred_boxes`: Boxes in `[cx, cy, w, h]` normalized to `[0, 1]`.

**DETRLiteDecoder output:** `{"pred_logits": (B, Q, C+1), "pred_boxes": (B, Q, 4)}`

- Class dim includes one background class at index `num_classes`.

**Notes:**
- `FPNHead` requires intermediate layers. These are auto-set by `DetectionHead()` if not already configured.
- `RTDETRDecoder` sets its own intermediate layers during construction.

---

## SegmentationHead

```python
head = fft.SegmentationHead(encoder, num_classes=5, head_type="linear", **kwargs)
```

**Parameters:**
- `encoder`: An `Encoder` instance or encoder directly.
- `num_classes` (int): Number of segmentation classes.
- `head_type` (str): One of `"linear"`, `"upernet"`, `"mask_transformer"`. Default: `"linear"`.
- `**kwargs`: Additional arguments passed to the decoder constructor (e.g., `output_size` for `UPerNetHead`).

**Head types:**

| Type | Class | Description |
|------|-------|-------------|
| `"linear"` | `LinearSegHead` | Per-patch 1×1 conv + bilinear upsample |
| `"upernet"` | `UPerNetHead` | UPerNet with pyramid pooling module |
| `"mask_transformer"` | `MaskTransformerHead` | Dot-product mask from learnable class queries |

**Output:** Logits of shape `(B, num_classes, H, W)`.

**Note:** `UPerNetHead` requires intermediate layers. These are auto-set by `SegmentationHead()` if not already configured.

---

## Dataset

```python
dataset = fft.Dataset.from_folder(
    "./data/",
    task="classification",
    transform=encoder.get_transform(),
    output_size=None,    # segmentation only: resize masks to this square size
)
```

`fft.Dataset` is an alias for `core.data.dataset.FFTDataset`.

**Class methods:**
- `from_folder(root, task, transform=None, **kwargs)`: Load from directory structure. For classification, expects one subfolder per class. For detection, expects `images/` and `annotations.json`. For segmentation, expects `images/` and `masks/`. Additional `**kwargs` (including `output_size`) are forwarded to the constructor.

**Constructor parameters:**
- `samples` (list[dict]): List of sample dicts.
- `task` (str): `"classification"`, `"detection"`, or `"segmentation"`.
- `transform`: Optional image transform.
- `output_size` (int | None): Segmentation only. When set, each mask is resized to `output_size × output_size` using `Image.NEAREST` in `__getitem__`. Pass the same value as the encoder input size to align mask and decoder output shapes. Propagated through `split()`.

**Instance methods:**
- `split(val_ratio=0.2, seed=42) -> (train_ds, val_ds)`: Split into train and validation sets. `output_size` is preserved in both subsets.
- `get_stats() -> dict`: Return dataset statistics (num_samples, num_classes, etc.).
- `FFTDataset.detection_collate_fn`: Static collate function for detection (handles variable-length box lists); passed automatically by `Trainer`.

**Properties:**
- `task` (str): Task type.
- `class_names` (list[str]): Class name list.
- `samples` (list[dict]): Raw sample list.

---

## Trainer

```python
trainer = fft.Trainer(
    decoder=head,
    train_dataset=dataset,
    val_dataset=None,                   # auto-split if None
    lr=1e-3,
    epochs=50,
    batch_size=32,
    scheduler="cosine",                 # "cosine" | "step" | "constant"
    warmup_epochs=5,
    augmentation="light",               # "none" | "light" | "heavy"
    early_stopping_patience=10,
    checkpoint_dir="./checkpoints",
    num_workers=4,
    accelerator="auto",
    devices="auto",
    val_ratio=0.2,
    training_mode="standard",           # "standard" | "deim"
    lora_rank=0,                        # 0 = disabled; >0 enables LoRA
)
```

**Methods:**
- `fit() -> dict`: Run training. Returns dict with `best_model_path`, `best_val_loss`, `epochs_trained`, and task-specific metrics (`val_acc` for classification, `val_map50` / `val_map` for detection).
- `save(path)`: Save decoder weights to `.pt` file.
- `load(path)`: Load decoder weights from `.pt` file.

**Training modes:**
- `"standard"`: Default training without augmentation scheduling.
- `"deim"`: Detection-specific. Wraps the training dataset with Mosaic augmentation (4-image grid), disabled automatically at `epochs * 0.5`. Also disables CDN in the final 2 epochs.

**LoRA:** When `lora_rank > 0`, `LoRALinear` adapters are injected into `attn.qkv` and `attn.proj` layers of the encoder before training. These parameters receive gradients; all other encoder parameters stay frozen.

---

## Inference

```python
results = fft.run_inference(decoder, image_paths)
```

Runs batch inference on a list of image file paths. Returns a list of prediction dicts.

- **Classification**: `{"class_idx": int, "class_name": str, "confidence": float}`
- **Detection**: `{"boxes": [...], "labels": [...], "scores": [...]}`
- **Segmentation**: `{"mask": ndarray}`

---

## Metrics

```python
metrics = fft.compute_metrics(predictions, targets, task="classification")
```

Computes task-specific metrics:
- **Classification**: `accuracy`, per-class accuracy
- **Detection**: `precision`, `recall`, `f1`, `mAP@50`
- **Segmentation**: `mIoU`, pixel accuracy, per-class IoU

---

## Weights

```python
fft.save_decoder_weights(decoder, "model.pt")
fft.load_decoder_weights(decoder, "model.pt")
```

Saves only the decoder parameters (encoder is frozen and not saved). The `.pt` file includes metadata about the decoder class, task, and architecture.

Typical file sizes: LinearProbe ~1 MB, MLPHead ~2 MB, RTDETRDecoder ~30 MB.

---

## Script Generation

```python
fft.generate_inference_script(
    decoder_class="LinearProbe",
    task="classification",
    encoder_name="dinov3_vitb16",
    num_classes=5,
    weights_path="model.pt",
    output_path="inference.py",
)
```

Generates a standalone Python inference script that can be distributed without the full library installed.

---

## Low-Level API

For more control, use the core library directly:

```python
from core.encoders import create_encoder, DINOv2Encoder, DINOv3Encoder
from core.decoders.rtdetr import RTDETRDecoder
from core.decoders.classification import LinearProbe, MLPHead
from core.decoders.segmentation import UPerNetHead
from core.data.dataset import FFTDataset
from core.data.formats import load_coco, load_voc, load_yolo, load_csv
from core.training.trainer import Trainer, DecoderLightningModule
from core.evaluation.inference import run_inference
from core.evaluation.metrics import compute_metrics
from core.export.weights import save_decoder_weights, load_decoder_weights

# Direct RTDETRDecoder construction
encoder = create_encoder("dinov2_vitb14")
head = RTDETRDecoder(
    encoder,
    num_classes=10,
    num_queries=100,          # or None for auto: min(max(C*10, 30), 300)
    num_decoder_layers=4,
    hidden_dim=256,
    num_heads=8,
    dim_feedforward=1024,
    max_gt_per_image=30,
)
```

---

## VLMDecoder

`VLMDecoder` is not part of the `sdk` module. Use the core API directly.

```python
from core.encoders import create_encoder
from core.decoders.vlm import VLMDecoder
from core.data.vqa_dataset import PetsVQADataset
from core.training.vlm_trainer import VLMTrainer
```

### VLMDecoder Constructor

```python
decoder = VLMDecoder(
    encoder,
    llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # default
    freeze_llm=True,
    lora_rank=0,
    pool_patches=2,
    load_in_4bit=False,
)
```

**Parameters:**
- `encoder`: A `DINOv2Encoder` or `DINOv3Encoder` instance (frozen).
- `llm_name` (str): HuggingFace model ID for the causal language model. Default: `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`. The cloud training script uses `"microsoft/Phi-3.5-mini-instruct"`.
- `freeze_llm` (bool): Freeze all LLM parameters (Stage 1 mode). Default: `True`.
- `lora_rank` (int): If >0, inject LoRA adapters into LLM `q_proj`/`v_proj` at construction. Default: `0`.
- `pool_patches` (int): Spatial pooling factor for DINOv2 patch tokens. `pool_patches=2` reduces a 16×16 grid to 8×8 = 64 visual tokens. Default: `2`.
- `load_in_4bit` (bool): Load the LLM in 4-bit NF4 quantization (requires `bitsandbytes`). When `True`, the LLM is placed on `cuda:0` automatically. Default: `False`.

**Properties:**
- `num_visual_tokens` (int): Number of visual tokens after pooling = `num_patches // pool_patches²`.
- `tokenizer`: The HuggingFace tokenizer for the LLM.
- `projector` (`nn.Sequential`): The 2-layer MLP projector (the only parameters trained in Stage 1).
- `llm`: The `AutoModelForCausalLM` language model.

**Methods:**
- `forward(features, input_ids, attention_mask, labels) -> dict`: Training forward pass. Returns `{"loss": scalar, "logits": (B, V+T, vocab_size)}`. Labels must be `-100` at question and padding positions.
- `generate(features, input_ids, attention_mask, max_new_tokens=128, **kwargs) -> list[str]`: Greedy decoding for inference. Returns a list of decoded answer strings.
- `enable_llm_lora(rank=8, target_modules=None) -> int`: Inject LoRA into LLM attention projections and set their `requires_grad=True`. Returns the number of layers patched.
- `trainable_parameters() -> list[Parameter]`: Returns all parameters with `requires_grad=True`.
- `num_trainable_params() -> int`: Count of trainable parameters.

**When `load_in_4bit=True`**: Do not call `decoder.to("cuda")` on the full decoder. Move only `decoder.projector` and the encoder to the target device. The quantized LLM is already on `cuda:0`.

### VLMTrainer

```python
trainer = VLMTrainer(
    decoder=decoder,
    train_dataset=train_ds,
    val_dataset=val_ds,
    lr=1e-3,
    epochs=3,
    batch_size=4,
    stage=1,                        # 1 = projector only; 2 = projector + LLM LoRA
    checkpoint_dir="./checkpoints",
    num_workers=0,
    warmup_epochs=1,
    early_stopping_patience=0,      # 0 = off
)
results = trainer.fit()
# results keys: best_val_loss, val_token_acc, epochs_trained, best_model_path
```

`VLMTrainer` is a standalone (non-Lightning) training loop for `VLMDecoder`.

**Parameters:**
- `stage` (int): `1` trains only the MLP projector. `2` trains projector + any LLM parameters with `requires_grad=True` (i.e., LoRA parameters after `enable_llm_lora()` is called).
- All other parameters (`lr`, `epochs`, `batch_size`, `warmup_epochs`, `early_stopping_patience`) behave the same as in `Trainer`.

**Return dict keys:**
- `best_val_loss`: Best validation loss over training.
- `val_token_acc`: Fraction of non-masked answer tokens predicted correctly at the best checkpoint epoch.
- `epochs_trained`: Number of epochs completed.
- `best_model_path`: Path to the saved checkpoint.

### PetsVQADataset

```python
from core.data.vqa_dataset import PetsVQADataset

dataset = PetsVQADataset(
    annotations_json="./oxford_pets/annotations.json",
    images_dir="./oxford_pets/images/",
    tokenizer=decoder.tokenizer,
    transform=encoder.get_transform(),
    max_length=256,
)
train_ds, val_ds = dataset.split(val_ratio=0.2, seed=42)
```

**Parameters:**
- `annotations_json`: Path to a COCO-format annotations JSON.
- `images_dir`: Directory containing image files.
- `tokenizer`: HuggingFace tokenizer (must match the LLM in `VLMDecoder`).
- `transform`: Image transform pipeline.
- `max_length` (int): Maximum token sequence length for padding/truncation. Default: `256`.

**`__getitem__` output:**
- `image`: Transformed image tensor.
- `input_ids`: Tokenized full sequence (question + answer), padded to `max_length`.
- `attention_mask`: Attention mask for `input_ids`.
- `labels`: LM labels; `-100` at question and padding positions, token ID at answer positions.
- `boxes`: Bounding boxes in `[cx, cy, w, h]` normalized format.
- `class_labels`: Class IDs corresponding to `boxes`.

QA templates (one selected randomly per `__getitem__`):
1. "What breed of animal is in this image?" → "This is a {breed}."
2. "What type of animal is shown here?" → "This is a {species}."
3. "Where is the {breed} located in the image?" → "The {breed} is in the {region} of the image."
4. "Describe what you see." → "I can see a {breed} in the {region} of the image."

Chat format uses Phi-3.5-mini native tokens: `<|user|>...<|end|>\n<|assistant|>\n{answer}<|end|>`.

**Static method:**
- `collate_fn(batch)`: Custom collate for VQA batches with variable-length box tensors.
