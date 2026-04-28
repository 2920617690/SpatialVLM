from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ProposalOutput:
    answer_logits: torch.Tensor
    evidence_query: torch.Tensor


class ProposalHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, answer_classes: int = 2) -> None:
        super().__init__()
        self.answer_head = nn.Linear(hidden_dim, answer_classes)
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, fused_context: torch.Tensor) -> ProposalOutput:
        return ProposalOutput(
            answer_logits=self.answer_head(fused_context),
            evidence_query=self.evidence_head(fused_context),
        )
