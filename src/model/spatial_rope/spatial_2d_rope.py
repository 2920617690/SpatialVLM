"""
Spatial 2D RoPE for Visual Tokens (方案 B / TODO 3)

核心思路：给 LLM 中的视觉 token 使用 2D 旋转位置编码，
替代默认的 1D 序列位置编码，让 LLM 的 self-attention
天然感知 patch 的 2D 空间邻接关系。

设计要点：
- 视觉 token 使用 2D RoPE：将 embedding 维度拆成两半，
  分别编码 row 和 col 位置
- 文本 token 保持 1D RoPE 不变
- 支持与现有 LLM（如 LLaMA）的 RoPE 实现兼容
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


def precompute_2d_rope_frequencies(
    dim: int,
    grid_height: int,
    grid_width: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 2D RoPE 的频率矩阵。

    将 embedding 维度拆成两半：
    - 前半部分编码 row（垂直位置）
    - 后半部分编码 col（水平位置）

    Args:
        dim: embedding 维度（必须是偶数）
        grid_height: patch grid 的高度
        grid_width: patch grid 的宽度
        base: RoPE 的基础频率
        device: 计算设备

    Returns:
        cos_cache: (grid_height * grid_width, dim) 余弦缓存
        sin_cache: (grid_height * grid_width, dim) 正弦缓存
    """
    assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"

    half_dim = dim // 2
    quarter_dim = dim // 4

    # 频率向量：每个维度对应一个频率
    freq_row = 1.0 / (base ** (torch.arange(0, quarter_dim, dtype=torch.float32, device=device) / quarter_dim))
    freq_col = 1.0 / (base ** (torch.arange(0, quarter_dim, dtype=torch.float32, device=device) / quarter_dim))

    # 生成 2D 网格坐标
    rows = torch.arange(grid_height, dtype=torch.float32, device=device)
    cols = torch.arange(grid_width, dtype=torch.float32, device=device)
    grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")

    # flatten 成序列：(grid_height * grid_width,)
    flat_row = grid_row.reshape(-1)
    flat_col = grid_col.reshape(-1)

    # 计算角度：(num_patches, quarter_dim)
    angles_row = torch.outer(flat_row, freq_row)
    angles_col = torch.outer(flat_col, freq_col)

    # 拼接：前半编码 row，后半编码 col
    # 每半部分内部再拆成 cos/sin 对
    # 最终形状：(num_patches, dim)
    angles = torch.cat([angles_row, angles_row, angles_col, angles_col], dim=-1)

    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)

    return cos_cache, sin_cache


def apply_2d_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 query 和 key 应用 2D 旋转位置编码。

    Args:
        query: (batch, num_heads, seq_len, head_dim)
        key: (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim) 余弦缓存
        sin: (seq_len, head_dim) 正弦缓存

    Returns:
        rotated_query: (batch, num_heads, seq_len, head_dim)
        rotated_key: (batch, num_heads, seq_len, head_dim)
    """
    # 标准 RoPE 旋转操作
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将 x 的后半部分取负并与前半部分交换。"""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    # 扩展 cos/sin 到 batch 和 head 维度
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    rotated_query = query * cos + rotate_half(query) * sin
    rotated_key = key * cos + rotate_half(key) * sin

    return rotated_query, rotated_key


class Spatial2DRoPE(nn.Module):
    """2D 旋转位置编码模块。

    为视觉 token 提供 2D 空间位置编码，
    同时兼容文本 token 的 1D 位置编码。
    """

    def __init__(
        self,
        head_dim: int,
        max_grid_size: int = 32,
        base: float = 10000.0,
    ):
        """
        Args:
            head_dim: 每个 attention head 的维度
            max_grid_size: 支持的最大 patch grid 尺寸
            base: RoPE 基础频率
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_grid_size = max_grid_size
        self.base = base

        # 预计算最大尺寸的频率缓存
        cos_cache, sin_cache = precompute_2d_rope_frequencies(
            dim=head_dim,
            grid_height=max_grid_size,
            grid_width=max_grid_size,
            base=base,
        )
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对视觉 token 的 query/key 应用 2D RoPE。

        Args:
            query: (batch, num_heads, num_visual_tokens, head_dim)
            key: (batch, num_heads, num_visual_tokens, head_dim)
            grid_height: 当前图像的 patch grid 高度
            grid_width: 当前图像的 patch grid 宽度

        Returns:
            rotated_query, rotated_key
        """
        num_patches = grid_height * grid_width

        # 重新计算当前分辨率的频率（如果不在缓存范围内）
        if grid_height > self.max_grid_size or grid_width > self.max_grid_size:
            cos, sin = precompute_2d_rope_frequencies(
                dim=self.head_dim,
                grid_height=grid_height,
                grid_width=grid_width,
                base=self.base,
                device=query.device,
            )
        else:
            # 从缓存中提取对应分辨率的子集
            cos, sin = self._extract_from_cache(grid_height, grid_width)

        return apply_2d_rotary_embedding(query, key, cos, sin)

    def _extract_from_cache(
        self, grid_height: int, grid_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从预计算的最大尺寸缓存中提取当前分辨率的子集。"""
        # 缓存是按 max_grid_size x max_grid_size 排列的
        # 需要提取 grid_height x grid_width 的子集
        indices = []
        for row in range(grid_height):
            for col in range(grid_width):
                indices.append(row * self.max_grid_size + col)

        indices_tensor = torch.tensor(indices, device=self.cos_cache.device)
        cos = self.cos_cache[indices_tensor]
        sin = self.sin_cache[indices_tensor]
        return cos, sin


class HybridPositionEmbedding(nn.Module):
    """混合位置编码：视觉 token 用 2D RoPE，文本 token 用 1D RoPE。

    在 LLM 的 attention 层中使用，替代原始的统一 1D RoPE。
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        max_grid_size: int = 32,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.spatial_rope = Spatial2DRoPE(head_dim, max_grid_size, base)

        # 1D RoPE 用于文本 token（标准实现）
        self.base = base
        self.max_seq_len = max_seq_len
        self._precompute_1d_rope(max_seq_len, head_dim, base)

    def _precompute_1d_rope(self, max_seq_len: int, head_dim: int, base: float):
        """预计算 1D RoPE 频率缓存。"""
        half_dim = head_dim // 2
        freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, freq)
        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer("text_cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("text_sin_cache", torch.sin(angles), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        num_visual_tokens: int,
        grid_height: int,
        grid_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对混合序列（视觉 + 文本）应用位置编码。

        假设输入序列的排列是：[visual_tokens | text_tokens]

        Args:
            query: (batch, num_heads, total_seq_len, head_dim)
            key: (batch, num_heads, total_seq_len, head_dim)
            num_visual_tokens: 视觉 token 的数量
            grid_height: patch grid 高度
            grid_width: patch grid 宽度

        Returns:
            rotated_query, rotated_key
        """
        # 拆分视觉和文本部分
        visual_query = query[:, :, :num_visual_tokens, :]
        text_query = query[:, :, num_visual_tokens:, :]
        visual_key = key[:, :, :num_visual_tokens, :]
        text_key = key[:, :, num_visual_tokens:, :]

        # 视觉部分用 2D RoPE
        rotated_visual_query, rotated_visual_key = self.spatial_rope(
            visual_query, visual_key, grid_height, grid_width
        )

        # 文本部分用 1D RoPE
        num_text_tokens = text_query.shape[2]
        text_cos = self.text_cos_cache[:num_text_tokens].unsqueeze(0).unsqueeze(0)
        text_sin = self.text_sin_cache[:num_text_tokens].unsqueeze(0).unsqueeze(0)
        rotated_text_query, rotated_text_key = apply_2d_rotary_embedding(
            text_query, text_key, text_cos.squeeze(0).squeeze(0), text_sin.squeeze(0).squeeze(0)
        )

        # 重新拼接
        final_query = torch.cat([rotated_visual_query, rotated_text_query], dim=2)
        final_key = torch.cat([rotated_visual_key, rotated_text_key], dim=2)

        return final_query, final_key
