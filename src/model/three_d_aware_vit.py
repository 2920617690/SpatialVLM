"""3D-aware Vision Transformer with parallel Depth Branch.

Extends a standard ViT by fusing RGB patch tokens with depth-aware tokens
through Gated Cross-Attention, producing 3D-aware visual representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .depth_branch import DepthBranch


class GatedCrossAttention(nn.Module):
    """Gated Cross-Attention for fusing RGB and Depth features.

    Uses cross-attention in both directions (RGB→Depth, Depth→RGB)
    with a learnable gate to control the fusion ratio.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize gated cross-attention.

        Args:
            dim: Feature dimension for both RGB and depth tokens.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.rgb_to_depth_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.depth_to_rgb_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Learnable fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.layer_norm_rgb = nn.LayerNorm(dim)
        self.layer_norm_depth = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)

        # Feed-forward after fusion
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, rgb_tokens: torch.Tensor, depth_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Fuse RGB and depth tokens via gated cross-attention.

        Args:
            rgb_tokens: RGB patch tokens of shape (batch, num_patches, dim).
            depth_tokens: Depth tokens of shape (batch, num_patches, dim).

        Returns:
            Fused 3D-aware tokens of shape (batch, num_patches, dim).
        """
        rgb_normed = self.layer_norm_rgb(rgb_tokens)
        depth_normed = self.layer_norm_depth(depth_tokens)

        # Bidirectional cross-attention
        rgb_attended, _ = self.rgb_to_depth_attn(
            query=rgb_normed, key=depth_normed, value=depth_normed
        )
        depth_attended, _ = self.depth_to_rgb_attn(
            query=depth_normed, key=rgb_normed, value=rgb_normed
        )

        # Residual connections
        rgb_enhanced = rgb_tokens + rgb_attended
        depth_enhanced = depth_tokens + depth_attended

        # Gated fusion
        gate_input = torch.cat([rgb_enhanced, depth_enhanced], dim=-1)
        gate = self.fusion_gate(gate_input)
        fused = gate * rgb_enhanced + (1 - gate) * depth_enhanced

        # Feed-forward with residual
        fused = fused + self.feed_forward(self.output_norm(fused))

        return fused


class ThreeDAwareViT(nn.Module):
    """3D-aware Vision Transformer.

    Combines a pretrained RGB ViT encoder with a parallel Depth Branch,
    fusing their outputs through Gated Cross-Attention to produce
    visual tokens that encode both appearance and 3D spatial structure.
    """

    def __init__(
        self,
        rgb_encoder: nn.Module,
        depth_branch: DepthBranch,
        rgb_dim: int = 1024,
        depth_dim: int = 1024,
        fused_dim: int = 1024,
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize the 3D-aware ViT.

        Args:
            rgb_encoder: Pretrained ViT encoder (e.g. SigLIP, CLIP ViT).
            depth_branch: Depth estimation branch (e.g. from Depth Anything V2).
            rgb_dim: Output dimension of the RGB encoder.
            depth_dim: Output dimension of the depth branch tokens.
            fused_dim: Dimension of the fused 3D-aware tokens.
            num_fusion_layers: Number of gated cross-attention fusion layers.
            num_heads: Number of attention heads in fusion layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.depth_branch = depth_branch

        # Projection layers to align dimensions before fusion
        self.rgb_projection = (
            nn.Linear(rgb_dim, fused_dim) if rgb_dim != fused_dim else nn.Identity()
        )
        self.depth_projection = (
            nn.Linear(depth_dim, fused_dim) if depth_dim != fused_dim else nn.Identity()
        )

        # Stacked gated cross-attention fusion layers
        self.fusion_layers = nn.ModuleList(
            [
                GatedCrossAttention(fused_dim, num_heads, dropout)
                for _ in range(num_fusion_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(fused_dim)

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract 3D-aware visual tokens from an RGB image.

        Args:
            image: RGB image tensor of shape (batch, 3, height, width).

        Returns:
            fused_tokens: 3D-aware visual tokens of shape (batch, num_patches, fused_dim).
            depth_map: Predicted depth map of shape (batch, 1, height, width).
        """
        # RGB encoder forward pass
        rgb_output = self.rgb_encoder(image)
        # Handle different ViT output formats
        if isinstance(rgb_output, dict):
            rgb_tokens = rgb_output.get("last_hidden_state", rgb_output.get("hidden_states", None))
            if rgb_tokens is None:
                raise ValueError("Cannot extract hidden states from RGB encoder output")
        elif isinstance(rgb_output, (tuple, list)):
            rgb_tokens = rgb_output[0]
        else:
            rgb_tokens = rgb_output

        # Remove CLS token if present (keep only patch tokens)
        if rgb_tokens.shape[1] > self.depth_branch.num_patches:
            rgb_tokens = rgb_tokens[:, 1:, :]

        # Depth branch forward pass
        depth_map, depth_tokens = self.depth_branch(image)

        # Project to common dimension
        rgb_projected = self.rgb_projection(rgb_tokens)
        depth_projected = self.depth_projection(depth_tokens)

        # Multi-layer gated cross-attention fusion
        fused = rgb_projected
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(fused, depth_projected)

        fused = self.output_norm(fused)

        return fused, depth_map

    def get_output_dim(self) -> int:
        """Return the output dimension of fused tokens."""
        return self.output_norm.normalized_shape[0]

    @classmethod
    def from_pretrained(
        cls,
        rgb_encoder_name: str = "google/siglip-so400m-patch14-384",
        depth_pretrained_path: Optional[str] = None,
        fused_dim: int = 1024,
        num_fusion_layers: int = 2,
        **kwargs,
    ) -> "ThreeDAwareViT":
        """Build a ThreeDAwareViT from pretrained components.

        Args:
            rgb_encoder_name: HuggingFace model name for the RGB ViT encoder.
            depth_pretrained_path: Path to Depth Anything V2 checkpoint (optional).
            fused_dim: Dimension of fused output tokens.
            num_fusion_layers: Number of fusion layers.
            **kwargs: Additional arguments forwarded to DepthBranch.

        Returns:
            Initialized ThreeDAwareViT with pretrained weights loaded.
        """
        from transformers import AutoModel

        rgb_encoder = AutoModel.from_pretrained(rgb_encoder_name)
        rgb_config = rgb_encoder.config
        rgb_dim = getattr(rgb_config, "hidden_size", 1024)

        depth_kwargs = {
            "image_size": kwargs.get("image_size", 384),
            "patch_size": kwargs.get("patch_size", 14),
            "encoder_dim": kwargs.get("depth_encoder_dim", 1024),
            "depth_token_dim": kwargs.get("depth_token_dim", 1024),
        }
        if depth_pretrained_path is not None:
            depth_branch = DepthBranch.from_pretrained(
                depth_pretrained_path, **depth_kwargs
            )
        else:
            depth_branch = DepthBranch(**depth_kwargs)

        depth_dim = depth_kwargs["depth_token_dim"]

        return cls(
            rgb_encoder=rgb_encoder,
            depth_branch=depth_branch,
            rgb_dim=rgb_dim,
            depth_dim=depth_dim,
            fused_dim=fused_dim,
            num_fusion_layers=num_fusion_layers,
        )
