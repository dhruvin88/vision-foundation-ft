"""VLM Decoder: DINOv2 patch tokens → MLP projector → Phi-3.5-mini.

Architecture (LLaVA 1.5):
    image → DINOv2 → patch_tokens [B, N, D]
                   → avg_pool (pool_patches × pool_patches)
                   → [B, N', D]
                   → MLP projector (D → llm_dim//2 → llm_dim)
                   → [visual_tokens | question_tokens] → Phi-3.5-mini → answer
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)


class VLMDecoder(nn.Module):
    """MLP projector + Phi-3.5-mini for visual question answering.

    Args:
        encoder: Frozen DINOv2 encoder.
        llm_name: HuggingFace model ID for the language model.
        freeze_llm: Whether to freeze LLM parameters (Stage 1).
        lora_rank: If >0, apply LoRA to LLM q_proj/v_proj at construction.
        pool_patches: Spatial pooling factor (2 → 16×16 → 8×8 = 64 visual tokens).
    """

    task = "vlm"

    def __init__(
        self,
        encoder: BaseEncoder,
        llm_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_llm: bool = True,
        lora_rank: int = 0,
        pool_patches: int = 2,
        load_in_4bit: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool_patches = pool_patches

        embed_dim = encoder.embed_dim

        logger.info("Loading tokenizer + LLM: %s", llm_name)
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Loading LLM in 4-bit NF4 quantization")

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16 if not load_in_4bit else None,
            device_map="cuda:0" if (load_in_4bit and torch.cuda.is_available()) else None,
        )

        llm_dim: int = self.llm.config.hidden_size
        logger.info(
            "LLM hidden_size=%d  encoder embed_dim=%d", llm_dim, embed_dim
        )

        # 2-layer MLP projector (LLaVA 1.5 style)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, llm_dim),
        )

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        if lora_rank > 0:
            self.enable_llm_lora(rank=lora_rank)

        logger.info(
            "VLMDecoder ready: %d projector params, LLM frozen=%s, "
            "visual_tokens=%d (pool=%d)",
            sum(p.numel() for p in self.projector.parameters()),
            freeze_llm,
            self.num_visual_tokens,
            pool_patches,
        )

    @property
    def num_visual_tokens(self) -> int:
        """Number of visual tokens produced after spatial pooling."""
        return self.encoder.num_patches // (self.pool_patches ** 2)

    def _encode_visual(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pool patch tokens and project to LLM token space.

        Args:
            features: Encoder output dict (must contain 'patch_tokens').

        Returns:
            Visual embeddings of shape (B, num_visual_tokens, llm_dim).
        """
        patch_tokens = features["patch_tokens"]  # (B, N, D)
        B, N, D = patch_tokens.shape
        h = w = int(N ** 0.5)

        # Spatial avg-pool: (B, N, D) → (B, D, H, W) → pool → (B, D, H', W')
        spatial = patch_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, h, w)
        pooled = F.avg_pool2d(spatial, kernel_size=self.pool_patches)    # (B, D, h', w')
        pooled = pooled.flatten(2).transpose(1, 2)                       # (B, N', D)

        # Project: cast to projector dtype, then to LLM compute dtype.
        # Use the embedding layer dtype (not raw param dtype, which is uint8 for 4-bit quant).
        proj_dtype = self.projector[0].weight.dtype
        llm_dtype = self.llm.get_input_embeddings().weight.dtype

        visual_tokens = self.projector(pooled.to(proj_dtype))   # (B, N', llm_dim)
        return visual_tokens.to(llm_dtype)

    def _prepare_inputs(
        self,
        features: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode visual tokens, embed text, and concatenate for LLM input.

        Returns:
            combined: (B, V+T, llm_dim) input embeddings.
            full_mask: (B, V+T) attention mask.
        """
        visual_tokens = self._encode_visual(features)   # (B, V, llm_dim)
        B, V, _ = visual_tokens.shape
        text_embeds = self.llm.get_input_embeddings()(input_ids).to(dtype=visual_tokens.dtype)
        combined = torch.cat([visual_tokens, text_embeds], dim=1)
        visual_mask = torch.ones(B, V, dtype=attention_mask.dtype, device=visual_tokens.device)
        full_mask = torch.cat([visual_mask, attention_mask], dim=1)
        return combined, full_mask

    def forward(
        self,
        features: dict[str, torch.Tensor],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            features: Encoder output dict from encoder.forward_features().
            input_ids: Tokenized question+answer of shape (B, T).
            attention_mask: Attention mask (B, T).
            labels: LM labels (B, T); -100 at question/padding positions.

        Returns:
            Dict with 'loss' (scalar) and 'logits' (B, V+T, vocab_size).
        """
        combined, full_mask = self._prepare_inputs(features, input_ids, attention_mask)
        B, V = combined.shape[0], self.num_visual_tokens

        # Prepend -100 labels for visual positions (ignored in loss)
        visual_labels = torch.full((B, V), -100, dtype=labels.dtype, device=combined.device)
        full_labels = torch.cat([visual_labels, labels], dim=1)     # (B, V+T)

        output = self.llm(
            inputs_embeds=combined,
            attention_mask=full_mask,
            labels=full_labels,
            use_cache=False,
        )
        return {"loss": output.loss, "logits": output.logits}

    @torch.no_grad()
    def generate(
        self,
        features: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> list[str]:
        """Greedy decode for inference.

        Args:
            features: Encoder output dict.
            input_ids: Question token ids (B, T).
            attention_mask: Question attention mask (B, T).
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments forwarded to llm.generate() (e.g. eos_token_id).

        Returns:
            List of decoded answer strings (length B).
        """
        combined, full_mask = self._prepare_inputs(features, input_ids, attention_mask)

        generate_kwargs = {
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }
        generated_ids = self.llm.generate(
            inputs_embeds=combined,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def enable_llm_lora(
        self,
        rank: int = 8,
        target_modules: list[str] | None = None,
    ) -> int:
        """Apply LoRA adapters to LLM attention projections and unfreeze them.

        Args:
            rank: Low-rank dimension.
            target_modules: Module name suffixes to patch (defaults to q_proj, v_proj).

        Returns:
            Number of layers patched.
        """
        from core.encoders.lora import apply_lora

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        n = apply_lora(self.llm, rank, float(rank), target_modules)
        for name, param in self.llm.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        logger.info("LLM LoRA enabled: rank=%d, %d layers patched", rank, n)
        return n

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return all parameters that require gradients (projector + LLM LoRA)."""
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters())
