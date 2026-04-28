from __future__ import annotations

import torch
import torch.nn as nn


class RelationVerifier(nn.Module):
    def __init__(self, hidden_dim: int = 256, verifier_classes: int = 3) -> None:
        super().__init__()
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.claim_proj = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, verifier_classes),
        )

    def _encode_crop(self, crop: torch.Tensor) -> torch.Tensor:
        features = self.crop_encoder(crop)
        return features.flatten(1)

    def forward(
        self,
        crop_a: torch.Tensor,
        crop_b: torch.Tensor,
        claim_features: torch.Tensor,
    ) -> torch.Tensor:
        feat_a = self._encode_crop(crop_a)
        feat_b = self._encode_crop(crop_b)
        claim = self.claim_proj(claim_features)
        joint = torch.cat([feat_a, feat_b, claim], dim=-1)
        return self.classifier(joint)
