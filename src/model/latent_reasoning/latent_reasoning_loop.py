"""
Latent Reasoning Loop — 隐空间迭代推理

核心思路：模拟人类"逐个看、然后串联"的空间推理过程。
人类看"苹果和香蕉哪个离杯子远"时，不是一次性看完整张图，
而是依次聚焦苹果、香蕉、杯子，把各自位置存入工作记忆，最后比较。

本模块在 ViT 的隐空间中实现这个过程：
- ViT 只跑一次，产出 patch features（frozen）
- 一组可学习的 latent tokens 在隐空间中迭代：
  - Cross-attend to text → 自主决定"接下来看什么"
  - Cross-attend to patches → 从图像对应区域读取视觉信息
  - Self-attend → 整合多轮积累的空间信息
- 不显式解析实体（不依赖"苹果"/"香蕉"等具体概念），泛化能力强
- 迭代次数可控，每轮 latent tokens 携带前轮记忆

替代方案 C（Spatial-Aware Cross Attention）：
- 方案 C 是单次 cross-attention + spatial bias → 浅层匹配
- Latent Reasoning Loop 是多轮迭代 + 记忆积累 → 深层推理
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def build_2d_sinusoidal_pos_embed(grid_height: int, grid_width: int, dim: int) -> torch.Tensor:
    """构建 2D 正弦位置编码。

    将 dim 拆成两半，前半编码 row，后半编码 col。
    生成固定的（不可学习的）位置信息，让 cross-attention 知道 patches 的空间位置。

    Returns:
        pos_embed: (1, grid_height * grid_width, dim)
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half_dim = dim // 2

    # 频率
    omega = torch.arange(half_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (2 * omega / half_dim))

    # Row embeddings
    rows = torch.arange(grid_height, dtype=torch.float32).unsqueeze(1)  # (H, 1)
    row_embed = torch.cat([
        torch.sin(rows * omega.unsqueeze(0)),
        torch.cos(rows * omega.unsqueeze(0)),
    ], dim=-1)  # (H, half_dim)

    # Col embeddings
    cols = torch.arange(grid_width, dtype=torch.float32).unsqueeze(1)  # (W, 1)
    col_embed = torch.cat([
        torch.sin(cols * omega.unsqueeze(0)),
        torch.cos(cols * omega.unsqueeze(0)),
    ], dim=-1)  # (W, half_dim)

    # 组合：每个 (row, col) 位置的 embedding = [row_embed | col_embed]
    pos = torch.zeros(grid_height, grid_width, dim)
    pos[:, :, :half_dim] = row_embed.unsqueeze(1).expand(-1, grid_width, -1)
    pos[:, :, half_dim:] = col_embed.unsqueeze(0).expand(grid_height, -1, -1)

    return pos.reshape(1, grid_height * grid_width, dim)


class ReasoningBlock(nn.Module):
    """一轮迭代推理。

    每轮 latent tokens 执行三步：
    1. Cross-Attn to Text → "接下来该关注什么"（从文本获取意图）
    2. Cross-Attn to Patches → "看到了什么"（从图像读取视觉信息）
    3. Self-Attn → "综合起来是什么"（整合多轮积累的信息）
    4. FFN → 非线性变换

    所有步骤都有 pre-norm + residual connection。
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Cross-Attention: latent → text
        self.norm_text_q = nn.LayerNorm(latent_dim)
        self.norm_text_kv = nn.LayerNorm(latent_dim)
        self.text_cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 2. Cross-Attention: latent → patches
        self.norm_visual_q = nn.LayerNorm(latent_dim)
        self.norm_visual_kv = nn.LayerNorm(latent_dim)
        self.visual_cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 3. Self-Attention: latent ↔ latent
        self.norm_self = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 4. FFN
        self.norm_ffn = nn.LayerNorm(latent_dim)
        mlp_hidden = int(latent_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latent: torch.Tensor,
        text_features: torch.Tensor,
        patch_features: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latent: (B, K, latent_dim) 当前 latent state
            text_features: (B, T, latent_dim) 文本特征（已投影）
            patch_features: (B, N, latent_dim) patch 特征（已投影 + pos embed）
            text_padding_mask: (B, T) True = padding

        Returns:
            latent: (B, K, latent_dim) 更新后的 latent state
        """
        # 1. Cross-attend to text: "接下来该看什么？"
        q = self.norm_text_q(latent)
        kv = self.norm_text_kv(text_features)
        latent = latent + self.text_cross_attn(
            q, kv, kv, key_padding_mask=text_padding_mask
        )[0]

        # 2. Cross-attend to patches: "看到了什么？"
        q = self.norm_visual_q(latent)
        kv = self.norm_visual_kv(patch_features)
        latent = latent + self.visual_cross_attn(q, kv, kv)[0]

        # 3. Self-attend: "综合起来是什么？"
        normed = self.norm_self(latent)
        latent = latent + self.self_attn(normed, normed, normed)[0]

        # 4. FFN
        latent = latent + self.ffn(self.norm_ffn(latent))

        return latent


class LatentReasoningLoop(nn.Module):
    """隐空间迭代推理模块。

    在 ViT patch features 上进行多轮迭代推理，
    不需要多次跑 ViT，只在隐空间中循环。

    使用方式：
    1. ViT 编码图像 → patch_features (B, 729, 1152)
    2. Text encoder 编码文本 → text_hidden_states (B, T, 4096)
    3. LatentReasoningLoop 迭代推理 → reasoning_tokens (B, K, 4096)
    4. reasoning_tokens 与 visual/text tokens 拼接送入 LLM
    """

    def __init__(
        self,
        visual_dim: int = 1152,
        text_dim: int = 4096,
        latent_dim: int = 512,
        num_latent_tokens: int = 8,
        num_iterations: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 4096,
        max_grid_size: int = 32,
        share_weights: bool = True,
    ):
        """
        Args:
            visual_dim: ViT 输出维度
            text_dim: 文本编码器输出维度
            latent_dim: 隐空间维度（瓶颈）
            num_latent_tokens: latent tokens 数量 K
            num_iterations: 迭代轮数 N
            num_heads: attention head 数量
            mlp_ratio: FFN 扩展比率
            dropout: dropout 率
            output_dim: 输出维度（通常 = LLM dim）
            max_grid_size: 最大 patch grid 尺寸（用于 pos embed 预计算）
            share_weights: 是否所有迭代共享同一个 ReasoningBlock
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.num_iterations = num_iterations
        self.share_weights = share_weights

        # 可学习的 latent tokens 初始状态
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latent_tokens, latent_dim) * 0.02)

        # 输入投影
        self.visual_proj = nn.Linear(visual_dim, latent_dim)
        self.text_proj = nn.Linear(text_dim, latent_dim)

        # 2D 正弦位置编码（固定，不可学习）
        pos_embed = build_2d_sinusoidal_pos_embed(max_grid_size, max_grid_size, latent_dim)
        self.register_buffer("pos_embed_cache", pos_embed)

        # 迭代编码：告诉模型当前是第几轮
        self.iteration_embed = nn.Parameter(torch.randn(num_iterations, 1, latent_dim) * 0.02)

        # Reasoning blocks
        if share_weights:
            self.reasoning_block = ReasoningBlock(latent_dim, num_heads, mlp_ratio, dropout)
        else:
            self.reasoning_blocks = nn.ModuleList([
                ReasoningBlock(latent_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_iterations)
            ])

        # 输出
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, output_dim)

    def _get_pos_embed(self, grid_height: int, grid_width: int) -> torch.Tensor:
        """获取指定 grid 尺寸的 2D 位置编码。"""
        if grid_height * grid_width <= self.pos_embed_cache.shape[1]:
            # 从预计算的 cache 中裁切
            # cache 是按 max_grid_size × max_grid_size 排列的
            max_w = int(math.sqrt(self.pos_embed_cache.shape[1]))
            pos = self.pos_embed_cache.view(1, max_w, max_w, -1)
            pos = pos[:, :grid_height, :grid_width, :]
            return pos.reshape(1, grid_height * grid_width, -1)
        else:
            # 动态生成
            return build_2d_sinusoidal_pos_embed(
                grid_height, grid_width, self.latent_dim
            ).to(self.latent_tokens.device)

    def forward(
        self,
        patch_features: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        grid_height: int = 27,
        grid_width: int = 27,
    ) -> torch.Tensor:
        """
        Args:
            patch_features: (B, N, visual_dim) ViT 原始输出
            text_hidden_states: (B, T, text_dim) 文本编码器输出
            text_padding_mask: (B, T) True = padding
            grid_height, grid_width: patch grid 尺寸

        Returns:
            reasoning_tokens: (B, K, output_dim) 推理结果 tokens
        """
        B = patch_features.shape[0]

        # 投影到 latent space
        patch_proj = self.visual_proj(patch_features)   # (B, N, latent_dim)
        text_proj = self.text_proj(text_hidden_states)   # (B, T, latent_dim)

        # 加 2D 位置编码到 patch features
        pos_embed = self._get_pos_embed(grid_height, grid_width)
        patch_proj = patch_proj + pos_embed.to(patch_proj.device)

        # 初始化 latent tokens
        latent = self.latent_tokens.expand(B, -1, -1)  # (B, K, latent_dim)

        # 迭代推理
        for i in range(self.num_iterations):
            # 加上迭代编码（让模型知道当前是第几轮）
            latent = latent + self.iteration_embed[i].unsqueeze(0)

            # 选择 reasoning block
            if self.share_weights:
                block = self.reasoning_block
            else:
                block = self.reasoning_blocks[i]

            # 一轮推理
            latent = block(latent, text_proj, patch_proj, text_padding_mask)

        # 输出投影到 LLM 空间
        reasoning_tokens = self.output_proj(self.output_norm(latent))

        return reasoning_tokens
