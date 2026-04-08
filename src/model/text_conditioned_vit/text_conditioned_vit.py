"""
Text-Conditioned ViT (方案 A / TODO 1)

核心思路：在 ViT 的中间层注入文本 token，让文本作为 spatial query
引导 ViT 的 attention 关注与问题相关的空间区域。

与 QA-ViT 的区别：ViT 不冻结，允许文本信号反向传播调整视觉编码。
与 ViLT 的区别：从预训练 ViT 初始化，不从头训练，数据效率更高。

设计要点：
- 文本 token 通过 cross attention 注入 ViT 的指定层
- 注入后 ViT 的 self-attention 中，image patches 可以 attend 到文本 token
- 文本 token 携带空间意图（如"左边"、"上方"），引导 attention pattern 变得 spatially-aware
"""

import torch
import torch.nn as nn
from typing import Optional, List


class TextInjectionLayer(nn.Module):
    """在 ViT 的某一层注入文本 token 的模块。

    文本 token 通过 cross attention 与 image patch tokens 交互，
    然后拼接到 patch 序列中参与后续的 self-attention。
    """

    def __init__(self, visual_dim: int, text_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        # 将文本 embedding 投影到视觉空间
        self.text_projector = nn.Linear(text_dim, visual_dim)

        # 文本 attend 视觉特征，提取空间相关信息
        self.text_to_visual_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_text = nn.LayerNorm(visual_dim)

        # 门控机制：控制文本信号注入的强度，训练初期接近 0 避免破坏预训练权重
        self.injection_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: (batch, num_patches, visual_dim) ViT 中间层的 patch embeddings
            text_tokens: (batch, seq_len, text_dim) 文本编码器输出的 token embeddings
            text_padding_mask: (batch, seq_len) 文本 padding mask

        Returns:
            injected_visual_tokens: (batch, num_patches, visual_dim) 注入文本信息后的 patch embeddings
        """
        # 投影文本到视觉空间
        projected_text = self.text_projector(text_tokens)

        # 文本 query attend 视觉 key/value，提取与文本相关的视觉上下文
        text_context, attention_weights = self.text_to_visual_attn(
            query=projected_text,
            key=visual_tokens,
            value=visual_tokens,
            key_padding_mask=None,
        )
        text_context = self.norm_text(text_context + projected_text)

        # 用门控的 text_context 调制视觉 tokens
        # 视觉 tokens attend 到 text_context，获取文本引导的空间偏置
        gate = torch.tanh(self.injection_gate)
        spatial_bias = self._compute_spatial_bias(visual_tokens, text_context, text_padding_mask)
        injected_visual_tokens = visual_tokens + gate * spatial_bias

        return injected_visual_tokens

    def _compute_spatial_bias(
        self,
        visual_tokens: torch.Tensor,
        text_context: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算文本引导的空间偏置。

        visual tokens attend to text context，
        让每个 patch 根据文本语义获得不同的调制信号。
        """
        # (batch, num_patches, visual_dim) @ (batch, visual_dim, text_len)
        # -> (batch, num_patches, text_len)
        attention_logits = torch.bmm(
            visual_tokens, text_context.transpose(1, 2)
        ) / (self.visual_dim ** 0.5)

        if text_padding_mask is not None:
            attention_logits = attention_logits.masked_fill(
                text_padding_mask.unsqueeze(1), float("-inf")
            )

        attention_probs = torch.softmax(attention_logits, dim=-1)

        # (batch, num_patches, text_len) @ (batch, text_len, visual_dim)
        # -> (batch, num_patches, visual_dim)
        spatial_bias = torch.bmm(attention_probs, text_context)
        return spatial_bias


class TextConditionedViT(nn.Module):
    """Text-Conditioned ViT：在预训练 ViT 的指定层注入文本 token。

    使用方式：
    1. 加载预训练 ViT（如 SigLIP、DINOv2）
    2. 在指定层插入 TextInjectionLayer
    3. 微调时 ViT 不冻结，允许文本信号调整视觉编码
    """

    def __init__(
        self,
        vit_encoder: nn.Module,
        text_dim: int = 4096,
        injection_layers: Optional[List[int]] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            vit_encoder: 预训练的 ViT 模型（需要暴露中间层 blocks）
            text_dim: 文本编码器的 embedding 维度
            injection_layers: 在哪些 ViT 层注入文本，默认在 1/3 和 2/3 处
            num_heads: 注入层的 attention head 数量
            dropout: dropout 比率
        """
        super().__init__()
        self.vit_encoder = vit_encoder

        # 获取 ViT 的层数和隐藏维度
        self.vit_blocks = self._get_vit_blocks()
        num_layers = len(self.vit_blocks)
        visual_dim = self._get_visual_dim()

        # 默认在 1/3 和 2/3 处注入
        if injection_layers is None:
            injection_layers = [num_layers // 3, 2 * num_layers // 3]
        self.injection_layer_indices = set(injection_layers)

        # 为每个注入点创建 TextInjectionLayer
        self.injection_modules = nn.ModuleDict({
            str(layer_idx): TextInjectionLayer(
                visual_dim=visual_dim,
                text_dim=text_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for layer_idx in injection_layers
        })

    def _get_vit_blocks(self) -> nn.ModuleList:
        """从 ViT 中提取 transformer blocks，兼容不同 ViT 实现。"""
        if hasattr(self.vit_encoder, "blocks"):
            return self.vit_encoder.blocks
        if hasattr(self.vit_encoder, "encoder") and hasattr(self.vit_encoder.encoder, "layers"):
            return self.vit_encoder.encoder.layers
        if hasattr(self.vit_encoder, "layers"):
            return self.vit_encoder.layers
        raise ValueError(
            "Cannot find transformer blocks in ViT encoder. "
            "Expected attributes: 'blocks', 'encoder.layers', or 'layers'"
        )

    def _get_visual_dim(self) -> int:
        """获取 ViT 的隐藏维度。"""
        first_block = self.vit_blocks[0]
        if hasattr(first_block, "norm1"):
            return first_block.norm1.normalized_shape[0]
        if hasattr(first_block, "ln_1"):
            return first_block.ln_1.normalized_shape[0]
        raise ValueError("Cannot determine visual dim from ViT blocks")

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, channels, height, width) 输入图像
            text_tokens: (batch, seq_len, text_dim) 文本编码器输出
            text_padding_mask: (batch, seq_len) 文本 padding mask

        Returns:
            visual_features: (batch, num_patches, visual_dim) 文本引导的视觉特征
        """
        # Patch embedding
        hidden_states = self._patch_embed(pixel_values)

        # 逐层前向传播，在指定层注入文本
        for layer_idx, block in enumerate(self.vit_blocks):
            hidden_states = block(hidden_states)

            if layer_idx in self.injection_layer_indices:
                injection_module = self.injection_modules[str(layer_idx)]
                hidden_states = injection_module(
                    visual_tokens=hidden_states,
                    text_tokens=text_tokens,
                    text_padding_mask=text_padding_mask,
                )

        return hidden_states

    def _patch_embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """调用 ViT 的 patch embedding 层。"""
        if hasattr(self.vit_encoder, "patch_embed"):
            return self.vit_encoder.patch_embed(pixel_values)
        if hasattr(self.vit_encoder, "embeddings"):
            return self.vit_encoder.embeddings(pixel_values)
        raise ValueError("Cannot find patch embedding in ViT encoder")
