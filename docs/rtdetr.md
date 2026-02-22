# RTDETRDecoder — Architecture & Design Guide

`RTDETRDecoder` (`core/decoders/rtdetr.py`) is the default detection head in this platform, inspired by RT-DETRv2. It replaces the earlier `DETRLiteDecoder` with a suite of improvements that together produce significantly better mAP at similar inference cost.

---

## Why RT-DETR over vanilla DETR?

| Problem with vanilla DETR | RT-DETR fix |
|---|---|
| Single-scale features — misses small objects | Multi-scale ViT + CNN feature fusion |
| Slow query convergence — 100+ epochs needed | Top-k proposal initialization (warm start) |
| Bilinear box regression diverges early | Iterative refinement per decoder layer |
| Cross-entropy with ε-weight for background | Varifocal loss — soft IoU-weighted targets |
| No training signal for easy negatives | Contrastive denoising (CDN) |
| Fixed 100 queries regardless of class count | Auto-scaled: `min(max(C×10, 30), 300)` |

---

## Forward Pass Overview

```
Raw image (B, 3, H, W)
       │
       ├──── ViT encoder (frozen) ──────────────────────────────────────────┐
       │     intermediate layers [n/4-1, n/2-1, 3n/4-1]                    │
       │     → 3 feature maps (B, D, 37, 37) each                          │
       │                                                                     │
       └──── SpatialPriorModule (CNN branch) ────────────────────────────── │
             → s0 (B, 32, ~130, ~130) stride 4                              │
             → s1 (B, 64, ~65,  ~65)  stride 8                             │
             → s2 (B, 64, ~33,  ~33)  stride 16                            │
                                                                             │
                         ChannelMapper (fuse ViT + CNN per scale)           │
                         adaptive-pool CNN to ViT grid, cat, Conv1×1        │
                         → 3 × (B, 256, H_i, W_i)                          │
                                                                             │
                         HybridEncoder                                       │
                         transformer on s2 (81 tokens) + FPN top-down      │
                         → enriched [s0_fused, s1_fused, s2_enc]           │
                                                                             │
                         Flatten + concat → memory (B, 1774, 256)          │
                                                                             │
                         ProposalNetwork                                     │
                         score all 1774 positions, pick top-k               │
                         → query_content (B, K, 256)                        │
                         → query_ref_pts (B, K, 4)   ← warm-start boxes    │
                                                                             │
                [training only] CDNQueryBuilder prepends noisy GT queries   │
                                                                             │
                         4 × RTDETRDecoderLayer                             │
                         self-attn → cross-attn → FFN                      │
                         + iterative box refinement                         │
                         → pred_logits (B, K, C)                            │
                         → pred_boxes  (B, K, 4) ∈ [0,1]                   │
```

For ViT-B/14 with a 518×518 input the three scales are:
- **s0**: 37×37 = 1 369 tokens (full ViT resolution)
- **s1**: 18×18 = 324 tokens
- **s2**: 9×9  = 81 tokens
- **Total memory**: **1 774** positions

---

## Component Deep-Dive

### 1. SpatialPriorModule

```
Input: (B, 3, H, W)

layer0: Conv(3→16,  stride 2) + BN + ReLU  →  H/2
layer1: Conv(16→32, stride 2) + BN + ReLU  →  H/4   ← feat_s0 (32 ch)
layer2: Conv(32→64, stride 2) + BN + ReLU  →  H/8   ← feat_s1 (64 ch)
layer3: Conv(64→64, stride 2) + BN + ReLU  →  H/16  ← feat_s2 (64 ch)
```

The ViT is trained on 518×518 pixel patches of size 14×14 — it naturally lacks fine local texture cues at high frequency. The CNN branch runs in parallel on the raw pixels and captures exactly those details. Each strided layer doubles the effective receptive field while halving the spatial size.

The three output maps are later adaptive-pooled to match the ViT grid, so no fixed input size is assumed.

---

### 2. ChannelMapper

```
For each of the 3 scales:
    cnn_feat  ──► adaptive_avg_pool2d(target_size)
    vit_feat  ──┐
                cat → (B, vit_dim + cnn_dim, H_i, W_i)
                    → Conv1×1 + BN + ReLU
                    → (B, 256, H_i, W_i)
```

Channel widths per scale:

| Scale | ViT dim | CNN dim | Concat | → Output |
|---|---|---|---|---|
| s0 (37×37) | 768 | 32 | 800 | 256 |
| s1 (18×18) | 768 | 64 | 832 | 256 |
| s2 (9×9)   | 768 | 64 | 832 | 256 |

The 1×1 convolution is a cheap learned projection — it lets the network decide how much weight to give the ViT vs CNN features at each scale.

---

### 3. HybridEncoder

```
s2 (coarsest, 81 tokens)
    → flatten → TransformerEncoderLayer (pre-norm, 8 heads)
    → reshape back to (B, 256, 9, 9) = s2_enc

s2_enc ──► upsample to s1 size
           + s1 → fpn_conv1 (Conv3×3 + BN + ReLU) = s1_fused

s1_fused ──► upsample to s0 size
             + s0 → fpn_conv0 (Conv3×3 + BN + ReLU) = s0_fused

Returns: [s0_fused, s1_fused, s2_enc]
```

The transformer is run only on the coarsest scale (81 positions) for efficiency — global context propagates downward through the FPN. Running it on all 1774 positions would cost ~22× more attention FLOPs.

The FPN top-down path fuses high-level semantics from s2 into s1 and s0, giving fine-grained features an understanding of what objects are present globally.

---

### 4. ProposalNetwork (top-k query initialization)

Vanilla DETR uses randomly initialized learnable queries, which take many epochs to converge. The ProposalNetwork replaces this with a warm start:

```
flat_feats (B, 1774, 256)
    │
    ├── score_head (Linear → C logits per position)
    │   → enc_logits (B, 1774, C)
    │
    └── bbox_head (MLP 256→256→4)
        → delta (B, 1774, 4)

enc_boxes = sigmoid(inv_sigmoid(anchors) + delta)
    ↑ refines grid-cell anchors with learned offsets

scores = enc_logits.sigmoid().max(-1)
topk_indices = scores.topk(K)

query_content = flat_feats[topk_indices]   ← feature at that position
query_ref_pts = enc_boxes[topk_indices]    ← predicted box at that position
```

**Anchors** are uniform grid cell centers across all three scales: `cx = (j+0.5)/W`, `cy = (i+0.5)/H`, `bw = 1/W`, `bh = 1/H`. They are built lazily and cached so the grid computation only happens once per unique input resolution.

The queries that start the decoder loop are therefore positions the encoder already thinks are likely to contain objects — not random noise. This typically saves 20–30 epochs of decoder warm-up.

---

### 5. RTDETRDecoderLayer (iterative box refinement)

Each of the 4 decoder layers:

```
queries (B, Q, 256), ref_pts (B, Q, 4), memory (B, 1774, 256)

Pre-norm self-attention:
    norm1(queries) → self_attn(q, q, q) → residual add
    [optional attn_mask blocks CDN↔main cross-talk]

Pre-norm cross-attention:
    norm2(queries) → cross_attn(q, memory, memory) → residual add

Pre-norm FFN:
    norm3(queries) → Linear(256→1024) → ReLU → Linear(1024→256) → residual add

Prediction heads:
    pred_logits = cls_head(queries)          (B, Q, C)  — raw logits, sigmoid outside
    delta       = bbox_head(queries)          (B, Q, 4)
    new_ref     = sigmoid(inv_sigmoid(ref_pts) + delta)  (B, Q, 4)

Returns: updated_queries, pred_logits, new_ref (w/ grad), new_ref.detach() (for next layer)
```

**Why pre-norm?** Placing LayerNorm before each sub-layer (rather than after) stabilises training at larger depths and learning rates.

**Why iterative refinement?** Each layer refines the previous layer's box prediction rather than predicting absolute coordinates from scratch. The formula `sigmoid(inv_sigmoid(ref) + delta)` ensures:
- `delta = 0` → box unchanged (identity shortcut)
- The output is always a valid box in `[0, 1]`
- Gradients flow back through both the delta and the reference point cleanly

`new_ref.detach()` is passed to the next layer so each layer learns to refine independently, without the pathological gradient flow that would arise if layer _i_'s box prediction were used as a differentiable input to layer _i+1_.

---

### 6. Contrastive Denoising (CDN)

CDN is a training-time technique that creates additional supervised query slots alongside the normal object queries. The insight: the decoder must learn to distinguish a slightly noisy GT box+label from a heavily corrupted one.

```
For each batch image with M ground-truth objects:

    G groups (G = max(1, 100 // max_gt_per_image))

    Each group has 2M slots:
        M positive queries:
            content  = label_embedding(gt_label, possibly flipped with p=0.5)
            ref_pts  = gt_box + small noise (σ = 0.1 × box_noise_scale)

        M negative queries:
            content  = label_embedding(random class)
            ref_pts  = gt_box + large noise (σ = 0.5 × box_noise_scale)

Total CDN slots = G × 2M  (prepended before the K main queries)
```

**Attention mask** ensures CDN queries only attend to their own group. This prevents the G groups from seeing each other's answers (which would be easy to copy) and prevents CDN from contaminating the main queries:

```
Query layout: [CDN group 0 (2M)] [CDN group 1 (2M)] ... [main (K)]

Mask (True = blocked):
    CDN[i] → CDN[j]:  blocked unless i and j are in the same group
    CDN → main:        always blocked
    main → CDN:        always blocked
    main → main:       always allowed
```

At inference, CDN is inactive (training flag is False), so there is zero overhead.

**Loss for CDN queries:**
- Positive: BCE(pred_class, gt_class) + 5 × L1(pred_box, gt_box) + 2 × GIoU
- Negative: BCE(pred_class, zeros) — push all class scores toward zero

---

### 7. Varifocal Loss (VFL)

Standard cross-entropy treats every matched positive equally regardless of prediction quality. VFL replaces this with soft, quality-weighted targets.

**Target construction** — for each query `q` matched to GT box `g`:
```
target[q, class_g] = IoU(pred_box_q, gt_box_g)   ← quality score in [0, 1]
target[q, other]   = 0
```
Unmatched queries have all-zero targets.

**Weight construction:**
```
weight = α · sigmoid(logit)^γ · (1 − target)   ← background queries
       + target                                   ← foreground queries
```
with `α = 0.75`, `γ = 2.0`.

This is the same focal modulation as Focal Loss for background tokens (downweights easy negatives), but foreground tokens get `weight = iou_score` — a higher-quality prediction gets a proportionally stronger gradient signal.

```python
loss = BCE_with_logits(pred_logits, targets, weight=weight, reduction='sum')
loss /= max(num_matched, 1)
```

The weight is computed under `torch.no_grad()` so it does not create a second-order gradient path.

---

### 8. Full Training Loss

```
L = L_vfl + 5·L_L1 + 2·L_GIoU          ← final decoder layer (with Hungarian match)
  + Σ_layers (same terms, same matches)   ← all 4 aux outputs
  + 0.5 · BCE(enc_logits, zeros)          ← encoder auxiliary
  + Σ_cdn_layers L_cdn                    ← CDN groups (all 4 layers)
```

**Hungarian matching** for the main loss uses cost:
```
cost = -sigmoid(pred_logit)[gt_class]   ← classification cost
     + 5 · L1(pred_box, gt_box)         ← L1 cost
     + 2 · (-GIoU(pred_box, gt_box))    ← GIoU cost (−1 to 1, higher = better)
```

The same matched indices are reused across all 4 aux outputs — computing Hungarian 4 separate times would give marginal improvement at substantial cost.

**Encoder auxiliary loss** penalises all 1 774 encoder positions for predicting any class, so the ProposalNetwork's background positions are actively suppressed.

---

## num_queries Auto-Scaling

```python
num_queries = min(max(num_classes * 10, 30), 300)
```

| num_classes | num_queries | rationale |
|---|---|---|
| 1 | 30 | floor: at least 30 slots even for single class |
| 3 | 30 | still floored |
| 5 | 50 | 5 × 10 |
| 10 | 100 | matches DETRLite default |
| 20 | 200 | scales with class count |
| ≥30 | 300 | ceiling: memory/speed cap |

More classes → more plausible object instances in a scene → more queries needed. The ceiling prevents memory blowout on datasets with many fine-grained categories.

---

## Output Dictionary

| Key | Shape | Present | Description |
|---|---|---|---|
| `pred_logits` | `(B, K, C)` | always | Final layer class logits (use `.sigmoid()` for scores) |
| `pred_boxes` | `(B, K, 4)` | always | Final layer boxes, cxcywh in `[0,1]` |
| `aux_outputs` | list of 4 dicts | always | All 4 decoder layers' predictions |
| `enc_outputs` | dict | always | Encoder-level logits + boxes over all 1774 positions |
| `cdn_outputs` | list of 4 dicts | training + GT only | CDN predictions per decoder layer |
| `cdn_meta` | dict | training + GT only | `{G, M, num_cdn}` |
| `cdn_gt_labels` | `(B, M_max)` | training + GT only | GT labels passed through for loss computation |
| `cdn_gt_boxes` | `(B, M_max, 4)` | training + GT only | GT boxes passed through for loss computation |

No background class in `pred_logits` — use `sigmoid()` not `softmax()`. The `enc_outputs` key is used by the trainer to route to `_detection_loss_rtdetr` instead of the DETRLite loss.

---

## Usage

```python
import sdk as fft

# RTDETRDecoder is the default — no head_type needed
encoder = fft.Encoder("dinov2_vitb14")
head = fft.DetectionHead(encoder, num_classes=10)

print(type(head))        # RTDETRDecoder
print(head.num_queries)  # 100  (10 × 10)
print(encoder.model.intermediate_layers)  # [2, 5, 8]  (set automatically)

# Explicit construction
from core.decoders.rtdetr import RTDETRDecoder
head = RTDETRDecoder(
    encoder,
    num_classes=10,
    num_queries=100,       # or None for auto
    num_decoder_layers=4,
    hidden_dim=256,
    num_heads=8,
    dim_feedforward=1024,
    max_gt_per_image=30,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
)

# Inference
head.eval()
out = head.predict(images)          # (B, 3, 518, 518)
scores = out["pred_logits"].sigmoid()   # (B, K, C)
boxes  = out["pred_boxes"]              # (B, K, 4) cxcywh
```

### CLI

```bash
# rtdetr is now the default for detection
python -m core.cli train \
    --encoder dinov2_vitb14 \
    --task detection \
    --decoder rtdetr \
    --data ./datasets/coco_subset \
    --num-classes 10 \
    --epochs 50

# To use the old DETRLite head instead
python -m core.cli train --decoder detr_lite ...
```

---

## Comparison to DETRLiteDecoder

| | DETRLiteDecoder | RTDETRDecoder |
|---|---|---|
| Features | Single-scale patch tokens | 3-scale ViT + CNN |
| Query init | Random learnable embeddings | Top-k from encoder proposals |
| Box prediction | Absolute sigmoid | Iterative refinement per layer |
| Classification loss | CrossEntropy + eos_coef=0.1 | Varifocal loss |
| Training tricks | None | CDN (contrastive denoising) |
| Convergence | ~50–100 epochs | ~20–30 epochs |
| `num_queries` | Fixed 100 | Auto-scaled |
| Background class | Yes (index `num_classes`) | No |
| Trainable params | ~3M | ~8M |
