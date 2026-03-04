"""RT-DETRv2-inspired detection decoder with multi-scale features,
varifocal loss, top-k query initialization, iterative box refinement,
and contrastive denoising (CDN).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.decoders.base import BaseDecoder
from core.encoders.base import BaseEncoder


def _inv_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))


class _MLP(nn.Module):
    """Simple MLP with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialPriorModule(nn.Module):
    """4-layer strided CNN producing multi-scale features from the raw image.

    Produces three feature maps at strides 4, 8, 16 with channel widths
    32, 64, 64 respectively.  For a 518×518 input the spatial sizes are
    approximately 130×130, 65×65, 33×33 before adaptive pooling.
    """

    def __init__(self) -> None:
        super().__init__()

        def _cbr(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.layer0 = _cbr(3, 16)   # stride 2
        self.layer1 = _cbr(16, 32)  # stride 4  → feat_s0 (32 ch)
        self.layer2 = _cbr(32, 64)  # stride 8  → feat_s1 (64 ch)
        self.layer3 = _cbr(64, 64)  # stride 16 → feat_s2 (64 ch)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.layer0(image)
        s0 = self.layer1(x)   # 32 ch, stride 4
        s1 = self.layer2(s0)  # 64 ch, stride 8
        s2 = self.layer3(s1)  # 64 ch, stride 16
        return s0, s1, s2


class ChannelMapper(nn.Module):
    """Per-scale projection: (vit_dim + cnn_dim) → hidden_dim."""

    def __init__(self, vit_dim: int, cnn_dims: list[int], hidden_dim: int = 256) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(vit_dim + cnn_d, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for cnn_d in cnn_dims
        ])

    def forward(
        self,
        vit_feats: list[torch.Tensor],
        cnn_feats: list[torch.Tensor],
        target_sizes: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        outputs = []
        for vit_f, cnn_f, target_size, proj in zip(
            vit_feats, cnn_feats, target_sizes, self.projections
        ):
            cnn_resized = F.adaptive_avg_pool2d(cnn_f, target_size)
            fused = torch.cat([vit_f, cnn_resized], dim=1)
            outputs.append(proj(fused))
        return outputs


class HybridEncoder(nn.Module):
    """Transformer on coarsest scale + FPN top-down fusion."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.transformer_enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.fpn_conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fpn_conv0 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            feats: [s0 (B,D,H0,W0), s1 (B,D,H1,W1), s2 (B,D,H2,W2)]
        Returns:
            [s0_fused, s1_fused, s2_enc]
        """
        s0, s1, s2 = feats
        B, D, H2, W2 = s2.shape

        # Transformer on coarsest scale
        s2_flat = s2.flatten(2).permute(0, 2, 1)  # (B, H2*W2, D)
        s2_enc_flat = self.transformer_enc(s2_flat)
        s2_enc = s2_enc_flat.permute(0, 2, 1).reshape(B, D, H2, W2)

        # FPN top-down: s2↑ fused into s1
        s1_fused = self.fpn_conv1(
            s1 + F.interpolate(s2_enc, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        )

        # FPN top-down: s1↑ fused into s0
        s0_fused = self.fpn_conv0(
            s0 + F.interpolate(s1_fused, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        )

        return [s0_fused, s1_fused, s2_enc]


class ProposalNetwork(nn.Module):
    """Selects top-k encoder positions as initial queries."""

    def __init__(self, hidden_dim: int, num_classes: int, num_queries: int) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.score_head = _MLP(hidden_dim, hidden_dim, num_classes, num_layers=3)
        self.bbox_head = _MLP(hidden_dim, hidden_dim, 4, num_layers=2)

    def forward(
        self,
        flat_feats: torch.Tensor,
        flat_anchors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            flat_feats:   (B, N, D)
            flat_anchors: (N, 4) cxcywh in [0,1]
        Returns:
            (query_content, query_ref_pts, enc_logits, enc_boxes)
        """
        enc_logits = self.score_head(flat_feats)  # (B, N, C)
        delta = self.bbox_head(flat_feats)         # (B, N, 4)

        # Iterative refinement from anchors
        enc_boxes = (_inv_sigmoid(flat_anchors.unsqueeze(0)) + delta).sigmoid()  # (B, N, 4)

        # Select top-k by max class score
        with torch.no_grad():
            scores = enc_logits.sigmoid().max(-1).values  # (B, N)
            topk_indices = scores.topk(self.num_queries, dim=-1).indices  # (B, K)

        # Gather top-k
        topk_idx_exp = topk_indices.unsqueeze(-1)
        query_content = flat_feats.gather(
            1, topk_idx_exp.expand(-1, -1, flat_feats.shape[-1])
        )  # (B, K, D)
        query_ref_pts = enc_boxes.gather(
            1, topk_idx_exp.expand(-1, -1, 4)
        )  # (B, K, 4)

        return query_content, query_ref_pts, enc_logits, enc_boxes


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward)
        self.w3 = nn.Linear(d_model, dim_feedforward)
        self.w2 = nn.Linear(dim_feedforward, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RTDETRDecoderLayer(nn.Module):
    """Pre-norm decoder layer: self-attn → cross-attn → FFN + iterative box refinement."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = SwiGLUFFN(hidden_dim, dim_feedforward, dropout)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim)
        self.cls_head = _MLP(hidden_dim, hidden_dim, num_classes, num_layers=3)
        self.bbox_head = _MLP(hidden_dim, hidden_dim, 4, num_layers=2)

    def forward(
        self,
        queries: torch.Tensor,
        ref_pts: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (updated_queries, pred_logits, pred_boxes, next_ref_pts_detached)
        """
        # Self-attention (pre-norm)
        q = self.norm1(queries)
        q2, _ = self.self_attn(q, q, q, attn_mask=attn_mask)
        queries = queries + q2

        # Cross-attention (pre-norm)
        q = self.norm2(queries)
        q2, _ = self.cross_attn(q, memory, memory)
        queries = queries + q2

        # FFN (pre-norm)
        q = self.norm3(queries)
        queries = queries + self.ffn(q)

        # Predict
        pred_logits = self.cls_head(queries)                        # (B, Q, C)
        delta = self.bbox_head(queries)                             # (B, Q, 4)
        new_ref = (_inv_sigmoid(ref_pts) + delta).sigmoid()        # (B, Q, 4)

        return queries, pred_logits, new_ref, new_ref.detach()


def _build_attn_mask(
    num_cdn: int, num_main: int, G: int, M: int, device: torch.device
) -> torch.Tensor:
    """Build CDN attention mask.

    True = blocked (ignored in attention).
    CDN queries within the same group attend to each other; all other
    cross-group and CDN↔main interactions are masked.
    """
    Q_total = num_cdn + num_main
    mask = torch.ones(Q_total, Q_total, dtype=torch.bool, device=device)

    # main → main: fully allowed
    mask[num_cdn:, num_cdn:] = False

    # CDN: within-group allowed (each group has 2M slots: pos_M + neg_M)
    group_size = 2 * M
    for g in range(G):
        start = g * group_size
        end = start + group_size
        mask[start:end, start:end] = False

    return mask


class CDNQueryBuilder:
    """Builds contrastive denoising queries for training.

    Not an nn.Module — a helper that borrows the decoder's label_embedding.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        max_gt_per_image: int,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
    ) -> None:
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.max_gt_per_image = max_gt_per_image
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.num_groups = max(1, 100 // max_gt_per_image)

    def build(
        self,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        label_embedding: nn.Embedding,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int] | None:
        """
        Args:
            gt_labels:       (B, M_max) — class indices, -1 for padding
            gt_boxes:        (B, M_max, 4) — cxcywh normalized, padded
            label_embedding: nn.Embedding(num_classes, hidden_dim)
        Returns:
            (cdn_queries, cdn_ref_pts, G, M) or None if no valid GT.
        """
        B = gt_labels.shape[0]
        device = gt_labels.device
        G = self.num_groups

        valid_counts = (gt_labels >= 0).sum(-1)  # (B,)
        M = int(valid_counts.max().item())
        if M == 0:
            return None
        M = min(M, self.max_gt_per_image)

        all_queries: list[torch.Tensor] = []
        all_ref_pts: list[torch.Tensor] = []

        for b in range(B):
            valid_mask = gt_labels[b] >= 0
            labels_b = gt_labels[b][valid_mask][:M]  # (<=M,)
            boxes_b = gt_boxes[b][valid_mask][:M]     # (<=M, 4)
            n_valid = labels_b.shape[0]

            # Pad to M
            if n_valid < M:
                pad_labels = labels_b.new_zeros(M - n_valid)
                labels_b = torch.cat([labels_b, pad_labels])
                pad_boxes = boxes_b.new_zeros(M - n_valid, 4)
                pad_boxes[:, :2] = 0.5
                pad_boxes[:, 2:] = 0.1
                boxes_b = torch.cat([boxes_b, pad_boxes])

            group_queries: list[torch.Tensor] = []
            group_ref: list[torch.Tensor] = []

            for _ in range(G):
                # Positive: small noise
                pos_boxes = (boxes_b + torch.randn_like(boxes_b) * 0.1 * self.box_noise_scale).clamp(0, 1)

                pos_labels = labels_b.clone()
                if self.label_noise_ratio > 0:
                    noise_mask = torch.rand(M, device=device) < self.label_noise_ratio
                    random_labels = torch.randint(0, self.num_classes, (M,), device=device)
                    pos_labels = torch.where(noise_mask, random_labels, pos_labels)
                pos_embeds = label_embedding(pos_labels)  # (M, D)

                # Negative: large noise
                neg_boxes = (boxes_b + torch.randn_like(boxes_b) * 0.5 * self.box_noise_scale).clamp(0, 1)
                neg_labels = torch.randint(0, self.num_classes, (M,), device=device)
                neg_embeds = label_embedding(neg_labels)  # (M, D)

                group_queries.append(torch.cat([pos_embeds, neg_embeds], dim=0))  # (2M, D)
                group_ref.append(torch.cat([pos_boxes, neg_boxes], dim=0))        # (2M, 4)

            all_queries.append(torch.cat(group_queries, dim=0))   # (G*2M, D)
            all_ref_pts.append(torch.cat(group_ref, dim=0))       # (G*2M, 4)

        cdn_queries = torch.stack(all_queries, dim=0)   # (B, G*2M, D)
        cdn_ref_pts = torch.stack(all_ref_pts, dim=0)  # (B, G*2M, 4)

        return cdn_queries, cdn_ref_pts, G, M


class RTDETRDecoder(BaseDecoder):
    """RT-DETRv2-inspired detection decoder.

    Improvements over DETRLiteDecoder:
    - Multi-scale ViT + CNN features via SpatialPriorModule + ChannelMapper
    - Hybrid encoder (transformer on coarsest scale + FPN top-down)
    - Top-k query initialization via ProposalNetwork
    - Iterative box refinement in each decoder layer
    - Contrastive denoising (CDN) during training
    - Varifocal classification loss
    - ``num_queries`` auto-scaled to ``min(max(num_classes*10, 30), 300)``

    Args:
        encoder: Frozen encoder (intermediate_layers is set automatically).
        num_classes: Number of object classes (no background class).
        num_queries: If None, set by ``min(max(num_classes*10, 30), 300)``.
        num_decoder_layers: Number of decoder layers (default 4).
        hidden_dim: Hidden dimension for decoder (default 256).
        num_heads: Number of attention heads (default 8).
        dim_feedforward: FFN inner dimension (default 1024).
        dropout: Dropout rate (default 0.0).
        max_gt_per_image: CDN max GT per image (default 30).
        label_noise_ratio: Fraction of CDN positive labels to flip (default 0.5).
        box_noise_scale: Scale of box noise for CDN (default 1.0).
    """

    task = "detection"

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        num_queries: int | None = None,
        num_decoder_layers: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        max_gt_per_image: int = 30,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
    ) -> None:
        super().__init__(encoder, num_classes)

        if num_queries is None:
            num_queries = min(max(num_classes * 10, 30), 300)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Auto-configure intermediate layers for 3-scale extraction
        n = encoder._num_blocks
        encoder.intermediate_layers = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1]

        # CNN branch
        self.spatial_prior = SpatialPriorModule()

        # Channel mapper: fuse (ViT + CNN) per scale → hidden_dim
        self.channel_mapper = ChannelMapper(
            vit_dim=encoder.embed_dim, cnn_dims=[32, 64, 64], hidden_dim=hidden_dim
        )

        # Hybrid encoder
        self.hybrid_encoder = HybridEncoder(hidden_dim=hidden_dim, num_heads=num_heads)

        # Proposal network
        self.proposal_net = ProposalNetwork(
            hidden_dim=hidden_dim, num_classes=num_classes, num_queries=num_queries
        )

        # Label embedding for CDN query content
        self.label_embedding = nn.Embedding(num_classes, hidden_dim)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            RTDETRDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_classes=num_classes,
            )
            for _ in range(num_decoder_layers)
        ])

        # CDN query builder
        self.cdn_builder = CDNQueryBuilder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            max_gt_per_image=max_gt_per_image,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
        )

        self.cdn_enabled: bool = True

        # Lazy anchor cache keyed by (scale_sizes tuple)
        self._anchors_cache: dict = {}

    def _build_anchors(
        self,
        scale_sizes: list[tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        key = (tuple(scale_sizes), device)
        if key not in self._anchors_cache:
            parts = []
            for h, w in scale_sizes:
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(h, dtype=torch.float32),
                    torch.arange(w, dtype=torch.float32),
                    indexing="ij",
                )
                cx = (grid_x + 0.5) / w
                cy = (grid_y + 0.5) / h
                bw = torch.full_like(cx, 1.0 / w)
                bh = torch.full_like(cy, 1.0 / h)
                parts.append(torch.stack([cx, cy, bw, bh], dim=-1).reshape(-1, 4))
            self._anchors_cache[key] = torch.cat(parts, dim=0).to(device)
        return self._anchors_cache[key]

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # --- 1. Get ViT intermediate features (3 scales) ---
        if "intermediate" in features:
            vit_feats_raw = features["intermediate"]  # list of ≥3 (B, D, h, w)
        else:
            sf = features["spatial_features"]
            vit_feats_raw = [sf, sf, sf]

        B = vit_feats_raw[0].shape[0]
        device = vit_feats_raw[0].device

        H0, W0 = vit_feats_raw[0].shape[-2:]
        target_sizes: list[tuple[int, int]] = [
            (H0, W0),
            (H0 // 2, W0 // 2),
            (H0 // 4, W0 // 4),
        ]

        # Pool each intermediate feature to its target scale
        vit_feats = [
            F.adaptive_avg_pool2d(vit_feats_raw[i], target_sizes[i])
            for i in range(3)
        ]

        # --- 2. CNN branch ---
        if "image" in features:
            cnn_s0, cnn_s1, cnn_s2 = self.spatial_prior(features["image"])
        else:
            cnn_s0 = torch.zeros(B, 32, *target_sizes[0], device=device)
            cnn_s1 = torch.zeros(B, 64, *target_sizes[1], device=device)
            cnn_s2 = torch.zeros(B, 64, *target_sizes[2], device=device)
        cnn_feats = [cnn_s0, cnn_s1, cnn_s2]

        # --- 3. Channel mapper ---
        fused = self.channel_mapper(vit_feats, cnn_feats, target_sizes)

        # --- 4. Hybrid encoder ---
        enriched = self.hybrid_encoder(fused)  # [s0, s1, s2]

        # --- 5. Flatten to memory (B, N_total, D) and build anchors ---
        memory = torch.cat(
            [f.flatten(2).permute(0, 2, 1) for f in enriched], dim=1
        )  # (B, N, D) where N = H0*W0 + (H0//2)*(W0//2) + (H0//4)*(W0//4)
        anchors = self._build_anchors(target_sizes, device)  # (N, 4)

        # --- 6. Proposal network → initial queries ---
        query_content, query_ref_pts, enc_logits, enc_boxes = self.proposal_net(
            memory, anchors
        )

        # --- 7. CDN (training only, when GT provided) ---
        attn_mask: torch.Tensor | None = None
        num_cdn = 0
        cdn_meta: dict | None = None

        if self.training and self.cdn_enabled and "gt_labels" in features and "gt_boxes" in features:
            cdn_result = self.cdn_builder.build(
                features["gt_labels"], features["gt_boxes"], self.label_embedding
            )
            if cdn_result is not None:
                cdn_queries, cdn_ref, G, M = cdn_result
                num_cdn = cdn_queries.shape[1]

                query_content = torch.cat([cdn_queries, query_content], dim=1)
                query_ref_pts = torch.cat([cdn_ref, query_ref_pts], dim=1)

                attn_mask = _build_attn_mask(num_cdn, self.num_queries, G, M, device)
                cdn_meta = {"G": G, "M": M, "num_cdn": num_cdn}

        # --- 8. Decoder loop ---
        layer_logits: list[torch.Tensor] = []
        layer_boxes: list[torch.Tensor] = []

        ref_pts = query_ref_pts
        queries = query_content

        for layer in self.decoder_layers:
            queries, pred_logits, pred_boxes, ref_pts = layer(
                queries, ref_pts, memory, attn_mask
            )
            layer_logits.append(pred_logits)
            layer_boxes.append(pred_boxes)

        # --- 9. Build output dict ---
        # All decoder layers go into aux_outputs; pred_logits/pred_boxes = last layer
        aux_outputs = [
            {
                "pred_logits": layer_logits[i][:, num_cdn:],
                "pred_boxes": layer_boxes[i][:, num_cdn:],
            }
            for i in range(len(self.decoder_layers))
        ]

        out: dict[str, object] = {
            "pred_logits": aux_outputs[-1]["pred_logits"],
            "pred_boxes": aux_outputs[-1]["pred_boxes"],
            "aux_outputs": aux_outputs,
            "enc_outputs": {
                "pred_logits": enc_logits,
                "pred_boxes": enc_boxes,
            },
        }

        if num_cdn > 0 and cdn_meta is not None:
            out["cdn_outputs"] = [
                {
                    "pred_logits": layer_logits[i][:, :num_cdn],
                    "pred_boxes": layer_boxes[i][:, :num_cdn],
                }
                for i in range(len(self.decoder_layers))
            ]
            out["cdn_meta"] = cdn_meta
            out["cdn_gt_labels"] = features["gt_labels"]
            out["cdn_gt_boxes"] = features["gt_boxes"]

        return out
