"""Depth Estimation Branch based on DPT (Dense Prediction Transformer) architecture.

Provides monocular depth estimation and patchified depth tokens
for integration with the 3D-aware ViT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DepthBranch(nn.Module):
    """Depth estimation branch based on DPT architecture.

    Extracts depth maps from RGB images and converts them into
    patchified depth tokens aligned with ViT patch grid.
    Supports initialization from Depth Anything V2 pretrained weights.
    """

    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 14,
        encoder_dim: int = 1024,
        decoder_dims: Tuple[int, ...] = (256, 512, 1024, 1024),
        depth_token_dim: int = 1024,
        num_encoder_layers: int = 4,
        num_heads: int = 8,
    ):
        """Initialize the depth estimation branch.

        Args:
            image_size: Input image resolution (square).
            patch_size: Patch size matching the RGB ViT.
            encoder_dim: Hidden dimension of the depth encoder.
            decoder_dims: Channel dimensions for each DPT decoder stage.
            depth_token_dim: Output dimension of each depth token.
            num_encoder_layers: Number of transformer layers in the depth encoder.
            num_heads: Number of attention heads in the depth encoder.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

        # Lightweight depth encoder (patch embedding + transformer layers)
        self.patch_embed = nn.Conv2d(
            3, encoder_dim, kernel_size=patch_size, stride=patch_size
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, encoder_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # DPT-style multi-scale decoder for dense depth prediction
        self.reassemble_layers = nn.ModuleList()
        for target_dim in decoder_dims:
            self.reassemble_layers.append(
                nn.Sequential(
                    nn.Linear(encoder_dim, target_dim),
                    nn.GELU(),
                )
            )

        # Fusion blocks that progressively upsample and merge features
        self.fusion_blocks = nn.ModuleList()
        for i in range(len(decoder_dims)):
            in_channels = decoder_dims[i]
            out_channels = decoder_dims[i - 1] if i > 0 else decoder_dims[0]
            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )
            )

        # Final depth prediction head (outputs single-channel depth map)
        self.depth_head = nn.Sequential(
            nn.Conv2d(decoder_dims[0], 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(),  # depth is non-negative
        )

        # Projection from encoder features to depth tokens
        self.depth_token_projection = nn.Sequential(
            nn.Linear(encoder_dim, depth_token_dim),
            nn.GELU(),
            nn.Linear(depth_token_dim, depth_token_dim),
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract depth map and depth tokens from an RGB image.

        Args:
            image: RGB image tensor of shape (batch, 3, height, width).

        Returns:
            depth_map: Dense depth prediction of shape (batch, 1, height, width).
            depth_tokens: Patchified depth tokens of shape (batch, num_patches, depth_token_dim).
        """
        batch_size = image.shape[0]

        # Patch embedding
        patch_features = self.patch_embed(image)  # (B, encoder_dim, H/P, W/P)
        grid_h, grid_w = patch_features.shape[2], patch_features.shape[3]
        patch_features = patch_features.flatten(2).transpose(1, 2)  # (B, N, encoder_dim)
        patch_features = patch_features + self.position_embedding[:, : patch_features.shape[1]]

        # Transformer encoder
        encoded_features = self.encoder(patch_features)  # (B, N, encoder_dim)

        # Multi-scale reassembly
        reassembled = []
        for layer in self.reassemble_layers:
            feat = layer(encoded_features)  # (B, N, target_dim)
            feat = feat.transpose(1, 2).reshape(
                batch_size, -1, grid_h, grid_w
            )  # (B, C, H/P, W/P)
            reassembled.append(feat)

        # Progressive fusion (coarse to fine)
        fused = reassembled[-1]
        for i in range(len(self.fusion_blocks) - 1, -1, -1):
            fused = self.fusion_blocks[i](fused)
            if i > 0:
                fused = F.interpolate(
                    fused, size=reassembled[i - 1].shape[2:], mode="bilinear", align_corners=False
                )
                fused = fused + reassembled[i - 1]

        # Dense depth prediction
        depth_map = self.depth_head(fused)  # (B, 1, H/P, W/P)
        depth_map = F.interpolate(
            depth_map,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Depth tokens (aligned with ViT patches)
        depth_tokens = self.depth_token_projection(encoded_features)  # (B, N, depth_token_dim)

        return depth_map, depth_tokens

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> "DepthBranch":
        """Load depth branch with pretrained weights (e.g. Depth Anything V2).

        Args:
            pretrained_path: Path to pretrained checkpoint.
            **kwargs: Additional constructor arguments.

        Returns:
            DepthBranch instance with loaded weights.
        """
        model = cls(**kwargs)
        state_dict = torch.load(pretrained_path, map_location="cpu")

        # Handle potential key mismatches from Depth Anything V2
        if "model" in state_dict:
            state_dict = state_dict["model"]

        compatible_keys = {}
        model_keys = set(model.state_dict().keys())
        for key, value in state_dict.items():
            if key in model_keys and model.state_dict()[key].shape == value.shape:
                compatible_keys[key] = value

        model.load_state_dict(compatible_keys, strict=False)
        loaded_count = len(compatible_keys)
        total_count = len(model_keys)
        print(
            f"DepthBranch: loaded {loaded_count}/{total_count} parameters from {pretrained_path}"
        )
        return model
