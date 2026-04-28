from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    patch_features: torch.Tensor
    global_image: torch.Tensor
    question_features: torch.Tensor
    fused_context: torch.Tensor


class SimpleBackboneAdapter(nn.Module):
    """Minimal backbone adapter for AVV skeleton experiments.

    The adapter is intentionally simple: it provides the interfaces AVV needs
    without locking the repo to a specific pretrained VLM.
    """

    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 224,
        hidden_dim: int = 256,
        question_dim: int = 256,
        vocab_size: int = 4096,
        num_patches: int = 196,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim // 2, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.question_embedding = nn.Embedding(vocab_size, question_dim)
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _encode_question(self, question_tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.question_embedding(question_tokens)
        pooled = embedded.mean(dim=1)
        return self.question_proj(pooled)

    def forward(self, images: torch.Tensor, question_tokens: torch.Tensor) -> BackboneOutput:
        feature_map = self.image_encoder(images)
        batch_size, channels, height, width = feature_map.shape
        patch_features = feature_map.flatten(2).transpose(1, 2)
        global_image = patch_features.mean(dim=1)
        question_features = self._encode_question(question_tokens)
        fused_context = self.fusion(torch.cat([global_image, question_features], dim=-1))
        return BackboneOutput(
            patch_features=patch_features,
            global_image=global_image,
            question_features=question_features,
            fused_context=fused_context,
        )
