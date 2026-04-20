"""
Text-Conditioned ViT with KV Injection（方案 A v2）— Qwen3.5 适配版

灵感来源：DFlash（2602.06036）在 decoder 中通过 KV injection 注入 target model 的
context feature，实现持续、不稀释的条件化。

核心思路：将文本特征编码为 compact context feature，然后在 ViT 每层 attention 中
直接注入为额外的 K/V entries。Visual patch queries 可以直接 attend 到文本 KV，
实现"第三人称"式引导——文本块不独立理解图像，只作为引导器持续提供空间意图条件。

Qwen3.5 ViT 适配要点：
- Qwen3.5 ViT 使用合并的 qkv 投影（不是分开的 q_proj/k_proj/v_proj）
- 输入是 2D packed format (seq_len, hidden_size)，不是 (batch, seq_len, hidden_size)
- 使用 cu_seqlens 做 variable-length packing
- 有 rotary_pos_emb 应用在 Q/K 上
- Attention forward 签名: (hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Callable


class TextContextEncoder(nn.Module):
    """将文本 embeddings 压缩为紧凑的 context feature。

    将高维文本 embeddings (2560) 压缩为低维 context (256)，所有 ViT 层共享同一份 context。
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
            text_tokens: (B, T, text_dim) 或 (total_T, text_dim)
        Returns:
            context: same shape with last dim = context_dim
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

        # Per-head gate 初始化为 0 → tanh(0) = 0 → 初始时不影响原始 ViT
        self.gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context: (total_T, context_dim) packed text context
        Returns:
            k: (total_T, visual_dim)
            v: (total_T, visual_dim)
            gate: (1, num_heads, 1, 1) 经 tanh 激活的 gate 值
        """
        k = self.k_proj(context)
        v = self.v_proj(context)
        gate = torch.tanh(self.gate)
        return k, v, gate


class TextConditionedViTKV(nn.Module):
    """DFlash 风格 KV Injection 的 Text-Conditioned ViT — 适配 Qwen3.5。

    通过 monkey-patch Qwen3_5VisionAttention 的 forward，在每层（或指定层）
    将文本 context feature 投影为额外的 K/V entries 并 concatenate 到视觉 K/V 上。

    Qwen3.5 ViT 特点：
    - 合并的 qkv 投影: self.qkv = nn.Linear(dim, dim*3)
    - 2D packed format: 输入 (seq_len, dim)，用 cu_seqlens 区分不同图像
    - Rotary position embedding 应用在 Q/K 上
    """

    def __init__(
        self,
        vision_model: nn.Module,
        text_dim: int = 2560,
        context_dim: int = 256,
        injection_layers: Optional[List[int]] = None,
        num_heads: int = 16,
    ):
        """
        Args:
            vision_model: Qwen3_5VisionModel 实例
            text_dim: 文本特征维度（= LLM hidden_size = 2560）
            context_dim: 压缩后的 context 维度
            injection_layers: 注入哪些层，None 表示所有 24 层
            num_heads: ViT attention head 数量（Qwen3.5 = 16）
        """
        super().__init__()
        self.vision_model = vision_model

        # Qwen3.5 ViT 的 blocks
        self.vit_blocks = vision_model.blocks
        num_layers = len(self.vit_blocks)
        visual_dim = self.vit_blocks[0].attn.dim
        self.visual_dim = visual_dim
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads

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
        self._injection_context: Dict[str, object] = {}

        # Monkey-patch attention 层
        self._original_forwards: Dict[int, Callable] = {}
        self._patch_attention_layers()

    def _patch_attention_layers(self) -> None:
        """Monkey-patch Qwen3_5VisionAttention 的 forward 以支持 KV injection。"""
        for layer_idx in self.injection_layer_indices:
            block = self.vit_blocks[layer_idx]
            attn = block.attn
            kv_head = self.kv_injection_heads[str(layer_idx)]

            self._original_forwards[layer_idx] = attn.forward
            attn.forward = self._create_patched_forward(attn, kv_head)

    def _create_patched_forward(
        self, original_attn: nn.Module, kv_head: KVInjectionHead
    ) -> Callable:
        """创建 patched Qwen3_5VisionAttention forward。

        Qwen3.5 ViT attention 的原始 forward 流程：
        1. qkv = self.qkv(hidden_states)  → (seq_len, dim*3)
        2. reshape + split → Q, K, V 各 (seq_len, num_heads, head_dim)
        3. apply rotary_pos_emb to Q, K
        4. reshape → (1, num_heads, seq_len, head_dim)
        5. attention (per-chunk via cu_seqlens 或 flash attention)
        6. proj output

        Patched 版本在 step 3 之后、step 4 之前，将 text KV concat 到视觉 KV 上。
        由于 cu_seqlens 的存在，需要按 chunk 分别处理。
        """
        holder = self._injection_context
        num_heads = self.num_heads
        head_dim = self.head_dim

        def patched_forward(
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            **kwargs,
        ) -> torch.Tensor:
            text_context = holder.get("text_context")
            text_cu_seqlens = holder.get("text_cu_seqlens")

            # 如果没有 text context，直接调用原始 forward
            if text_context is None:
                return original_attn.forward(
                    hidden_states, cu_seqlens, rotary_pos_emb,
                    position_embeddings=position_embeddings, **kwargs,
                )

            seq_length = hidden_states.shape[0]

            # Step 1-2: QKV 投影 + split
            qkv = original_attn.qkv(hidden_states)
            query_states, key_states, value_states = (
                qkv.reshape(seq_length, 3, num_heads, head_dim)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )
            # 各自 shape: (seq_len, num_heads, head_dim)

            # Step 3: 应用 rotary position embedding
            if position_embeddings is not None:
                from transformers.models.qwen3_5.modeling_qwen3_5 import (
                    apply_rotary_pos_emb_vision,
                )
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb_vision(
                    query_states, key_states, cos, sin
                )

            # Step 4: 投影 text context 为 K/V
            k_text, v_text, gate = kv_head(text_context)
            # k_text, v_text: (total_T, visual_dim)
            k_text = k_text.view(-1, num_heads, head_dim)  # (total_T, heads, head_dim)
            v_text = v_text.view(-1, num_heads, head_dim)

            # Step 5: 按 cu_seqlens 分 chunk，每个 chunk 独立做 attention
            visual_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            num_chunks = len(visual_lengths)

            # text 的 cu_seqlens
            if text_cu_seqlens is not None:
                text_lengths = (text_cu_seqlens[1:] - text_cu_seqlens[:-1]).tolist()
            else:
                # 均分 text tokens 给每个 chunk，余数分配给最后一个 chunk
                total_text = text_context.shape[0]
                text_per_chunk = total_text // max(num_chunks, 1)
                remainder = total_text - text_per_chunk * num_chunks
                text_lengths = [text_per_chunk] * num_chunks
                if remainder > 0 and num_chunks > 0:
                    text_lengths[-1] += remainder

            # Split visual Q/K/V by chunks
            visual_q_chunks = torch.split(query_states, visual_lengths, dim=0)
            visual_k_chunks = torch.split(key_states, visual_lengths, dim=0)
            visual_v_chunks = torch.split(value_states, visual_lengths, dim=0)

            # Split text K/V by chunks
            text_k_chunks = torch.split(k_text, text_lengths, dim=0)
            text_v_chunks = torch.split(v_text, text_lengths, dim=0)

            attn_outputs = []
            # gate: (1, num_heads, 1, 1) → (1, num_heads, 1) for broadcasting with (T_i, num_heads, head_dim)
            gate_broadcast = gate.squeeze(0).squeeze(-1).unsqueeze(0)  # (1, num_heads, 1)

            for chunk_idx in range(num_chunks):
                chunk_q = visual_q_chunks[chunk_idx]   # (N_i, heads, head_dim)
                chunk_k = visual_k_chunks[chunk_idx]
                chunk_v = visual_v_chunks[chunk_idx]
                chunk_k_text = text_k_chunks[chunk_idx]  # (T_i, heads, head_dim)
                chunk_v_text = text_v_chunks[chunk_idx]

                # Gate text KV: broadcast (1, num_heads, 1) * (T_i, num_heads, head_dim)
                chunk_k_text = gate_broadcast * chunk_k_text
                chunk_v_text = gate_broadcast * chunk_v_text

                # Concat: [visual_KV | text_KV]
                combined_k = torch.cat([chunk_k, chunk_k_text], dim=0)
                combined_v = torch.cat([chunk_v, chunk_v_text], dim=0)

                # Reshape for attention: (1, heads, seq, head_dim)
                n_vis = chunk_q.shape[0]
                n_total = combined_k.shape[0]

                chunk_q = chunk_q.transpose(0, 1).unsqueeze(0)  # (1, heads, N_i, hd)
                combined_k = combined_k.transpose(0, 1).unsqueeze(0)  # (1, heads, N_i+T_i, hd)
                combined_v = combined_v.transpose(0, 1).unsqueeze(0)

                # Scaled dot-product attention
                chunk_out = F.scaled_dot_product_attention(
                    chunk_q, combined_k, combined_v,
                    attn_mask=None,
                    dropout_p=0.0 if not original_attn.training else original_attn.attention_dropout,
                    is_causal=False,
                )
                # (1, heads, N_i, head_dim) → (N_i, heads, head_dim) → (N_i, dim)
                chunk_out = chunk_out.squeeze(0).transpose(0, 1)
                attn_outputs.append(chunk_out)

            # Concat all chunks back
            attn_output = torch.cat(attn_outputs, dim=0)  # (seq_len, heads, head_dim)
            attn_output = attn_output.reshape(seq_length, -1).contiguous()

            # Output projection
            attn_output = original_attn.proj(attn_output)
            return attn_output

        return patched_forward

    def restore_attention_layers(self) -> None:
        """恢复所有 patched attention 层的原始 forward。"""
        for layer_idx, original_forward in self._original_forwards.items():
            block = self.vit_blocks[layer_idx]
            block.attn.forward = original_forward
        self._original_forwards.clear()

    def set_text_context(
        self,
        text_features: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        text_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """设置文本 context，供 monkey-patched attention 层读取。

        Args:
            text_features: (B, T, text_dim) 文本特征
            text_padding_mask: (B, T) True = padding
            text_cu_seqlens: (num_chunks+1,) 文本的 cumulative sequence lengths
        """
        batch_size = text_features.shape[0]

        # 编码为 compact context
        text_context = self.text_context_encoder(text_features)  # (B, T, context_dim)

        # 对 padding 位置置零
        if text_padding_mask is not None:
            text_context = text_context.masked_fill(
                text_padding_mask.unsqueeze(-1), 0.0
            )

        # Pack 成 2D format: (total_T, context_dim)
        if text_padding_mask is not None:
            # 去掉 padding tokens，只保留有效 tokens
            valid_tokens = []
            lengths = []
            for b in range(batch_size):
                valid_mask = ~text_padding_mask[b]
                valid_tokens.append(text_context[b][valid_mask])
                lengths.append(valid_mask.sum().item())
            packed_context = torch.cat(valid_tokens, dim=0)
            # 构建 text cu_seqlens
            text_cu_seqlens = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=text_features.device
            )
            for b in range(batch_size):
                text_cu_seqlens[b + 1] = text_cu_seqlens[b] + lengths[b]
        else:
            # 无 padding，直接 flatten
            packed_context = text_context.reshape(-1, text_context.shape[-1])
            seq_len = text_features.shape[1]
            text_cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, seq_len,
                dtype=torch.int32, device=text_features.device,
            )

        self._injection_context["text_context"] = packed_context
        self._injection_context["text_cu_seqlens"] = text_cu_seqlens

    def clear_text_context(self) -> None:
        """清空文本 context，避免状态泄漏。"""
        self._injection_context.clear()