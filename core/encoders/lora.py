"""Low-Rank Adaptation (LoRA) for frozen linear layers."""

from __future__ import annotations

import math

import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable low-rank A and B matrices.

    Output = W*x + scale * B(A(x)),  scale = alpha / rank.
    B initialized to zeros so LoRA starts as identity.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.linear = linear  # stays frozen
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        lora_out = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
        return self.linear(x) + self.scale * lora_out.to(x.dtype)


def apply_lora(
    model: nn.Module, rank: int, alpha: float, target_modules: list[str]
) -> int:
    """Replace Linear layers whose dotted name ends with a target suffix.

    Returns count of replaced layers.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        for suffix in target_modules:
            if name.endswith(suffix) and isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, rank, alpha))
                replaced += 1
                break
    return replaced
