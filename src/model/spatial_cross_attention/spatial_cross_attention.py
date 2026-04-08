"""
Spatial-Aware Cross Attention (方案 C / TODO 4)

核心思路：在 LLM 与 ViT 之间的 cross attention 中加入 spatial bias，
从文本 query 中提取空间意图（如"左边"、"上方"、"之间"），
引导 attention 偏向对应的空间区域。

与传统 cross attention 的区别：
- 传统：attention_score = Q_text @ K_visual.T
- 本方案：attention_score = Q_text @ K_visual.T + spatial_bias(text, patch_positions)

spatial_bias 由两部分组成：
1. Spatial Intent Extractor：从文本中提取空间意图向量
2. Position-Intent Matcher：将空间意图与 patch 的 2D 位置匹配，生成 bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SpatialIntentExtractor(nn.Module):
    """从文本 hidden states 中提取空间意图向量。

    空间意图表示文本 query 关注的空间区域特征，
    如方向（左/右/上/下）、距离（近/远）、关系（之间/旁边）等。
    """

    def __init__(self, text_dim: int, spatial_intent_dim: int = 64):
        """
        Args:
            text_dim: 文本 hidden state 的维度
            spatial_intent_dim: 空间意图向量的维度
        """
        super().__init__()
        self.intent_projector = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.GELU(),
            nn.Linear(text_dim // 2, spatial_intent_dim),
        )

        # 可学习的空间意图 query，用于从文本序列中聚合空间信息
        self.spatial_query = nn.Parameter(torch.randn(1, 1, text_dim) * 0.02)
        self.query_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_hidden_states: (batch, seq_len, text_dim)
            text_padding_mask: (batch, seq_len) True 表示 padding

        Returns:
            spatial_intent: (batch, spatial_intent_dim) 空间意图向量
        """
        batch_size = text_hidden_states.shape[0]

        # 用可学习 query 从文本中聚合空间信息
        spatial_query = self.spatial_query.expand(batch_size, -1, -1)
        aggregated, _ = self.query_attention(
            query=spatial_query,
            key=text_hidden_states,
            value=text_hidden_states,
            key_padding_mask=text_padding_mask,
        )

        # 投影到空间意图空间
        spatial_intent = self.intent_projector(aggregated.squeeze(1))
        return spatial_intent


class PositionIntentMatcher(nn.Module):
    """将空间意图与 patch 的 2D 位置匹配，生成 spatial bias。

    对每个 patch 位置，根据空间意图计算一个 bias 值，
    表示该位置与文本空间意图的匹配程度。
    """

    def __init__(self, spatial_intent_dim: int = 64, position_encoding_dim: int = 32):
        """
        Args:
            spatial_intent_dim: 空间意图向量的维度
            position_encoding_dim: 位置编码的中间维度
        """
        super().__init__()

        # 将 2D 坐标编码为高维向量
        self.position_encoder = nn.Sequential(
            nn.Linear(2, position_encoding_dim),
            nn.GELU(),
            nn.Linear(position_encoding_dim, spatial_intent_dim),
        )

        # 匹配层：计算空间意图与位置编码的兼容性
        self.match_layer = nn.Sequential(
            nn.Linear(spatial_intent_dim * 2, spatial_intent_dim),
            nn.GELU(),
            nn.Linear(spatial_intent_dim, 1),
        )

    def forward(
        self,
        spatial_intent: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> torch.Tensor:
        """
        Args:
            spatial_intent: (batch, spatial_intent_dim) 空间意图向量
            grid_height: patch grid 高度
            grid_width: patch grid 宽度

        Returns:
            spatial_bias: (batch, 1, grid_height * grid_width) 每个 patch 的空间偏置
        """
        batch_size = spatial_intent.shape[0]
        device = spatial_intent.device

        # 生成归一化的 2D 坐标网格 [0, 1]
        rows = torch.linspace(0, 1, grid_height, device=device)
        cols = torch.linspace(0, 1, grid_width, device=device)
        grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")
        positions = torch.stack([grid_row.reshape(-1), grid_col.reshape(-1)], dim=-1)
        # (num_patches, 2)

        # 编码位置
        position_encoding = self.position_encoder(positions)
        # (num_patches, spatial_intent_dim)

        # 扩展并拼接
        num_patches = grid_height * grid_width
        intent_expanded = spatial_intent.unsqueeze(1).expand(-1, num_patches, -1)
        position_expanded = position_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([intent_expanded, position_expanded], dim=-1)
        # (batch, num_patches, spatial_intent_dim * 2)

        # 计算匹配分数
        spatial_bias = self.match_layer(combined).squeeze(-1)
        # (batch, num_patches)

        return spatial_bias.unsqueeze(1)


class SpatialAwareCrossAttention(nn.Module):
    """空间感知的 Cross Attention 模块。

    在标准 cross attention 的基础上，加入从文本提取的空间偏置，
    引导 attention 关注与文本空间意图匹配的视觉区域。

    attention_score = Q_text @ K_visual.T / sqrt(d) + lambda * spatial_bias
    """

    def __init__(
        self,
        text_dim: int,
        visual_dim: int,
        num_heads: int = 8,
        spatial_intent_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            text_dim: LLM hidden state 维度
            visual_dim: ViT 输出的视觉特征维度
            num_heads: attention head 数量
            spatial_intent_dim: 空间意图向量维度
            dropout: dropout 比率
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        assert text_dim % num_heads == 0

        # 标准 cross attention 的 Q/K/V 投影
        self.query_proj = nn.Linear(text_dim, text_dim)
        self.key_proj = nn.Linear(visual_dim, text_dim)
        self.value_proj = nn.Linear(visual_dim, text_dim)
        self.output_proj = nn.Linear(text_dim, text_dim)

        # 空间感知组件
        self.spatial_intent_extractor = SpatialIntentExtractor(text_dim, spatial_intent_dim)
        self.position_intent_matcher = PositionIntentMatcher(spatial_intent_dim)

        # 空间偏置的强度系数（可学习）
        self.spatial_bias_scale = nn.Parameter(torch.tensor(0.1))

        self.attention_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(text_dim)

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        visual_features: torch.Tensor,
        grid_height: int,
        grid_width: int,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_hidden_states: (batch, text_seq_len, text_dim) LLM 的 hidden states
            visual_features: (batch, num_patches, visual_dim) ViT 输出
            grid_height: patch grid 高度
            grid_width: patch grid 宽度
            text_padding_mask: (batch, text_seq_len) 文本 padding mask

        Returns:
            output: (batch, text_seq_len, text_dim) cross attention 输出
            attention_weights: (batch, num_heads, text_seq_len, num_patches) attention 权重
        """
        batch_size, text_len, _ = text_hidden_states.shape
        num_patches = visual_features.shape[1]

        # 标准 Q/K/V 投影
        query = self.query_proj(text_hidden_states)
        key = self.key_proj(visual_features)
        value = self.value_proj(visual_features)

        # reshape for multi-head attention
        query = query.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        # 标准 attention score
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (batch, num_heads, text_len, num_patches)

        # 提取空间意图并计算 spatial bias
        spatial_intent = self.spatial_intent_extractor(
            text_hidden_states, text_padding_mask
        )
        spatial_bias = self.position_intent_matcher(
            spatial_intent, grid_height, grid_width
        )
        # spatial_bias: (batch, 1, num_patches) -> 广播到 (batch, num_heads, text_len, num_patches)

        # 加入 spatial bias
        attention_scores = attention_scores + self.spatial_bias_scale * spatial_bias.unsqueeze(2)

        # softmax + dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # 加权求和
        context = torch.matmul(attention_weights, value)
        # (batch, num_heads, text_len, head_dim)

        context = context.transpose(1, 2).contiguous().view(batch_size, text_len, -1)
        output = self.output_proj(context)

        # 残差连接 + LayerNorm
        output = self.layer_norm(output + text_hidden_states)

        return output, attention_weights
