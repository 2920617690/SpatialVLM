from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PointerOutput:
    boxes_a: torch.Tensor
    boxes_b: torch.Tensor


class EvidencePointerHead(nn.Module):
    """Predicts normalized evidence boxes for two object regions."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8),
        )

    @staticmethod
    def _normalize_boxes(boxes: torch.Tensor) -> torch.Tensor:
        boxes = torch.sigmoid(boxes)
        x1y1 = boxes[..., :2]
        x2y2 = boxes[..., 2:]
        mins = torch.minimum(x1y1, x2y2)
        maxs = torch.maximum(x1y1, x2y2)
        return torch.cat([mins, maxs], dim=-1)

    def forward(
        self,
        patch_features: torch.Tensor,
        question_features: torch.Tensor,
        evidence_query: torch.Tensor,
    ) -> PointerOutput:
        del patch_features
        joint = torch.cat([question_features, evidence_query], dim=-1)
        hidden = self.query_proj(joint)
        boxes = self.box_head(hidden)
        boxes_a = self._normalize_boxes(boxes[:, :4])
        boxes_b = self._normalize_boxes(boxes[:, 4:])
        return PointerOutput(boxes_a=boxes_a, boxes_b=boxes_b)
