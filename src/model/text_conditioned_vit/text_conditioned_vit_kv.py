"""
Text-Conditioned ViT with KV Injection（方案 A v2）

灵感来源：DFlash（2602.06036）在 decoder 中通过 KV injection 注入 target model 的
context feature，实现持续、不稀释的条件化。

核心思路：将文本特征编码为 compact context feature，然后在 ViT 每层 attention 中
直接注入为额外的 K/V entries。Visual patch queries 可以直接 attend 到文本 KV，
实现"第三人称"式引导——文本块不独立理解图像，只作为引导器持续提供空间意图条件。

与 v1（cross-attention + gate）的区别：
- v1 在 2 个注入点添加 additive spatial bias，信号在层间被稀释
- v2 在每层 attention 的 KV 中 concatenate 文本 entries，信号持续且直接

与 DFlash 的类比：
- DFlash：target LLM hidden features → fuse → inject into drafter's KV cache
- 本模块：text encoder features → compress → inject into ViT's KV cache
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import types


class TextContextEncoder(nn.Module):
    """将文本 embeddings 压缩为紧凑的 context feature。

    类比 DFlash 中从 target model 多层 hidden states 融合为 compact context feature 的过程。
    这里将高维文本 embeddings (4096) 压缩为低维 context (256)，所有 ViT 层共享同一份 context。
    """

    def __init__(self, text_dim: int, context_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(text_dim, context_dim * 2),
            nn.GELU(),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens: (B, T, text_dim)
        Returns:
            context: (B, T, context_dim)
        """
        return self.encoder(text_tokens)


class KVInjectionHead(nn.Module):
    """每层 ViT attention 的 KV 投影头。

    将 compact context feature 投影为该层 attention 的 K 和 V entries。
    Per-head gate 初始化为 0，保证初始时模型行为等价于原始 ViT。
    """

    def __init__(self, context_dim: int, visual_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads

        self.k_proj = nn.Linear(context_dim, visual_dim)
        self.v_proj = nn.Linear(context_dim, visual_dim)

        # Per-head gate: (1, num_heads, 1, 1)，初始化为 0
        # 训练时通过 tanh 激活，范围 [-1, 1]
        self.gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context: (B, T, context_dim)
        Returns:
            k: (B, T, visual_dim)
            v: (B, T, visual_dim)
            gate: (1, num_heads, 1, 1) 经 tanh 激活的 gate 值
        """
        k = self.k_proj(context)
        v = self.v_proj(context)
        gate = torch.tanh(self.gate)
        return k, v, gate


class TextConditionedViTKV(nn.Module):
    """DFlash 风格 KV Injection 的 Text-Conditioned ViT。

    通过 monkey-patch SigLIP attention 层的 forward，在每层（或指定层）
    将文本 context feature 投影为额外的 K/V entries 并 concatenate 到视觉 K/V 上。

    使用方式：
    1. 加载预训练 ViT（SigLIP）
    2. 创建本模块（自动 monkey-patch attention 层）
    3. forward 时传入 pixel_values + text_tokens
    4. 输出与原始 ViT 形状完全一致
    """

    def __init__(
        self,
        vit_encoder: nn.Module,
        text_dim: int = 4096,
        context_dim: int = 256,
        injection_layers: Optional[List[int]] = None,
        num_heads: int = 16,
    ):
        """
        Args:
            vit_encoder: 预训练 SigLIP ViT
            text_dim: 文本编码器输出维度
            context_dim: 压缩后的 context 维度
            injection_layers: 注入哪些层，None 表示所有层
            num_heads: ViT attention head 数量
        """
        super().__init__()
        self.vit_encoder = vit_encoder

        # 获取 ViT 结构信息
        self.vit_layers = self._get_vit_layers()
        num_layers = len(self.vit_layers)
        visual_dim = self._get_visual_dim()
        self.visual_dim = visual_dim
        self.num_heads = num_heads

        # 确定注入层：默认所有层
        if injection_layers is None:
            injection_layers = list(range(num_layers))
        self.injection_layer_indices = set(injection_layers)

        # 文本上下文编码器（共享，只计算一次）
        self.text_context_encoder = TextContextEncoder(text_dim, context_dim)

        # 每个注入层的 KV 投影头
        self.kv_injection_heads = nn.ModuleDict(
            {
                str(i): KVInjectionHead(context_dim, visual_dim, num_heads)
                for i in injection_layers
            }
        )

        # Holder dict：forward 时存入 text context，monkey-patched attention 读取
        self._text_context_holder: Dict[str, torch.Tensor] = {}

        # Monkey-patch attention 层
        self._original_forwards: Dict[int, callable] = {}
        self._patch_attention_layers()

    def _get_vit_layers(self) -> nn.ModuleList:
        """提取 ViT encoder layers，兼容 SigLIP 结构。"""
        # SigLIP: encoder.layers
        if hasattr(self.vit_encoder, "encoder") and hasattr(
            self.vit_encoder.encoder, "layers"
        ):
            return self.vit_encoder.encoder.layers
        # timm 风格: blocks
        if hasattr(self.vit_encoder, "blocks"):
            return self.vit_encoder.blocks
        # 直接 layers
        if hasattr(self.vit_encoder, "layers"):
            return self.vit_encoder.layers
        raise ValueError(
            "Cannot find transformer layers in ViT encoder. "
            "Expected: 'encoder.layers', 'blocks', or 'layers'"
        )

    def _get_visual_dim(self) -> int:
        """获取 ViT 隐藏维度。"""
        first_layer = self.vit_layers[0]
        # SigLIP: layer_norm1
        if hasattr(first_layer, "layer_norm1"):
            return first_layer.layer_norm1.normalized_shape[0]
        # timm: norm1
        if hasattr(first_layer, "norm1"):
            return first_layer.norm1.normalized_shape[0]
        if hasattr(first_layer, "ln_1"):
            return first_layer.ln_1.normalized_shape[0]
        raise ValueError("Cannot determine visual dim from ViT layers")

    def _patch_attention_layers(self) -> None:
        """Monkey-patch SigLIP attention layers 以支持 KV injection。"""
        for layer_idx in self.injection_layer_indices:
            layer = self.vit_layers[layer_idx]
            attn = layer.self_attn
            kv_head = self.kv_injection_heads[str(layer_idx)]

            # 保存原始 forward
            self._original_forwards[layer_idx] = attn.forward

            # 创建 patched forward
            patched_forward = self._create_patched_forward(attn, kv_head)
            attn.forward = patched_forward

    def _create_patched_forward(
        self, original_attn: nn.Module, kv_head: KVInjectionHead
    ):
        """创建一个 patched attention forward，支持 KV injection。

        当 holder 中有 text_context 时，将文本 KV concatenate 到视觉 KV 上。
        当 holder 为空时，退化为标准 self-attention。
        """
        holder = self._text_context_holder
        num_heads = self.num_heads
        head_dim = self.visual_dim // num_heads

        def patched_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            B, N, D = hidden_states.shape

            # 计算视觉 Q/K/V
            queries = original_attn.q_proj(hidden_states)
            keys = original_attn.k_proj(hidden_states)
            values = original_attn.v_proj(hidden_states)

            text_context = holder.get("text_context")

            if text_context is not None:
                T = text_context.shape[1]

                # 投影文本 context 为 K/V
                k_text, v_text, gate = kv_head(text_context)

                # Reshape to multi-head: (B, heads, seq, head_dim)
                queries = queries.view(B, N, num_heads, head_dim).transpose(1, 2)
                keys = keys.view(B, N, num_heads, head_dim).transpose(1, 2)
                values = values.view(B, N, num_heads, head_dim).transpose(1, 2)
                k_text = k_text.view(B, T, num_heads, head_dim).transpose(1, 2)
                v_text = v_text.view(B, T, num_heads, head_dim).transpose(1, 2)

                # Gate text KV
                k_text = gate * k_text
                v_text = gate * v_text

                # Concatenate: [visual_KV | text_KV]
                keys = torch.cat([keys, k_text], dim=2)  # (B, heads, N+T, head_dim)
                values = torch.cat([values, v_text], dim=2)

                # 构建扩展的 attention mask
                extended_mask = None
                if attention_mask is not None:
                    # 原始 mask: (B, 1, N, N) 或 (B, heads, N, N)
                    # 扩展为 (B, 1, N, N+T)：text 部分全部可 attend
                    text_mask = torch.zeros(
                        B, 1, N, T, dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    extended_mask = torch.cat([attention_mask, text_mask], dim=-1)

                # 处理文本 padding mask
                text_padding_mask = holder.get("text_padding_mask")
                if text_padding_mask is not None:
                    # text_padding_mask: (B, T)，True 表示 padding
                    # 转为 attention mask 格式: padding 位置设为 -inf
                    padding_bias = text_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
                    padding_bias = padding_bias.expand(-1, -1, N, -1)  # (B, 1, N, T)
                    padding_bias = padding_bias.float().masked_fill(padding_bias.bool(), float("-inf"))

                    if extended_mask is not None:
                        # 已有视觉 mask，只修改 text 部分
                        extended_mask = torch.cat(
                            [extended_mask[:, :, :, :N], extended_mask[:, :, :, N:] + padding_bias],
                            dim=-1,
                        )
                    else:
                        # 没有视觉 mask，创建新的: 视觉部分全 0 + text padding
                        visual_part = torch.zeros(
                            B, 1, N, N, dtype=padding_bias.dtype, device=padding_bias.device
                        )
                        extended_mask = torch.cat([visual_part, padding_bias], dim=-1)

                # Scaled dot-product attention（PyTorch 2.0+，天然支持 Q/KV 不同长度）
                attn_output = F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    attn_mask=extended_mask,
                    dropout_p=original_attn.dropout if original_attn.training else 0.0,
                )

                # Reshape back: (B, heads, N, head_dim) → (B, N, D)
                attn_output = attn_output.transpose(1, 2).reshape(B, N, D).contiguous()
            else:
                # 无文本 context → 标准 self-attention
                queries = queries.view(B, N, num_heads, head_dim).transpose(1, 2)
                keys = keys.view(B, N, num_heads, head_dim).transpose(1, 2)
                values = values.view(B, N, num_heads, head_dim).transpose(1, 2)

                attn_output = F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    attn_mask=attention_mask,
                    dropout_p=original_attn.dropout if original_attn.training else 0.0,
                )
                attn_output = attn_output.transpose(1, 2).reshape(B, N, D).contiguous()

            attn_output = original_attn.out_proj(attn_output)
            return attn_output, None

        return patched_forward

    def restore_attention_layers(self) -> None:
        """恢复所有 patched attention 层的原始 forward。"""
        for layer_idx, original_forward in self._original_forwards.items():
            layer = self.vit_layers[layer_idx]
            layer.self_attn.forward = original_forward
        self._original_forwards.clear()

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) 输入图像
            text_tokens: (B, T, text_dim) 文本编码器输出
            text_padding_mask: (B, T) True 表示 padding 位置

        Returns:
            visual_features: (B, num_patches, visual_dim)
        """
        # 1. 编码文本为 compact context（一次性，所有层共享）
        text_context = self.text_context_encoder(text_tokens)

        # 对 padding 位置置零，避免其影响 KV 投影
        if text_padding_mask is not None:
            text_context = text_context.masked_fill(
                text_padding_mask.unsqueeze(-1), 0.0
            )

        # 2. 存入 holder（monkey-patched attention 层会读取）
        self._text_context_holder["text_context"] = text_context
        self._text_context_holder["text_padding_mask"] = text_padding_mask

        try:
            # 3. 正常调用 ViT forward（内部 attention 已被 patch）
            visual_outputs = self.vit_encoder(pixel_values)

            # 兼容不同 ViT 输出格式
            if hasattr(visual_outputs, "last_hidden_state"):
                hidden_states = visual_outputs.last_hidden_state
            elif isinstance(visual_outputs, torch.Tensor):
                hidden_states = visual_outputs
            else:
                hidden_states = visual_outputs[0]
        finally:
            # 确保 holder 被清空，避免状态泄漏
            self._text_context_holder.clear()

        return hidden_states

    def forward_from_embeddings(
        self,
        patch_embeddings: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """从预计算的 patch embeddings 开始 forward（跳过 patch embed 层）。

        用于预训练时在 embedding 级别做 masking：先获取 patch embeddings，
        替换被遮挡位置为 mask token，再送入 ViT transformer 层。

        Args:
            patch_embeddings: (B, N, visual_dim) 已处理的 patch embeddings
            text_tokens: (B, T, text_dim) 文本编码器输出
            text_padding_mask: (B, T) True 表示 padding 位置

        Returns:
            visual_features: (B, N, visual_dim)
        """
        # 编码文本为 compact context
        text_context = self.text_context_encoder(text_tokens)
        if text_padding_mask is not None:
            text_context = text_context.masked_fill(
                text_padding_mask.unsqueeze(-1), 0.0
            )

        # 存入 holder
        self._text_context_holder["text_context"] = text_context
        self._text_context_holder["text_padding_mask"] = text_padding_mask

        try:
            hidden_states = patch_embeddings
            for layer in self.vit_layers:
                # SigLIP encoder layer 返回 tensor（不是 tuple）
                layer_output = layer(hidden_states, attention_mask=None)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output

            # 应用 post-layernorm（如果存在）
            if hasattr(self.vit_encoder, "post_layernorm"):
                hidden_states = self.vit_encoder.post_layernorm(hidden_states)
        finally:
            self._text_context_holder.clear()

        return hidden_states
