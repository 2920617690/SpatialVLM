"""
Spatial VLM：整合三个方案的完整模型。

Pipeline:
1. Text Encoder (frozen LLM 前几层 or 独立编码器) 编码文本
2. Text-Conditioned ViT (方案 A)：文本引导的视觉编码
3. Spatial 2D RoPE (方案 B)：视觉 token 使用 2D 位置编码
4. Spatial-Aware Cross Attention (方案 C)：空间感知的视觉-语言对齐
5. LLM Decoder 生成回答

方案 B 的集成说明：
HybridPositionEmbedding 需要替换 LLM 内部每一层 attention 的 RoPE。
由于不同 LLM 实现（LLaMA、Qwen 等）的 attention 层结构差异很大，
本模块提供 `patch_llm_with_spatial_rope()` 方法，通过 monkey-patch
的方式将 LLM 的 rotary embedding 替换为 HybridPositionEmbedding。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from functools import partial

from src.model.text_conditioned_vit.text_conditioned_vit import TextConditionedViT
from src.model.spatial_rope.spatial_2d_rope import HybridPositionEmbedding
from src.model.spatial_cross_attention.spatial_cross_attention import SpatialAwareCrossAttention


class SpatialVLM(nn.Module):
    """空间感知的 Vision-Language Model。

    通过三个互补的机制增强 VLM 的空间推理能力：
    - 方案 A：Text-Conditioned ViT，让视觉编码被文本意图引导
    - 方案 B：2D RoPE，让 LLM 感知视觉 token 的 2D 空间位置
    - 方案 C：Spatial-Aware Cross Attention，用空间意图引导视觉-语言对齐
    """

    def __init__(
        self,
        vit_encoder: nn.Module,
        llm_decoder: nn.Module,
        text_encoder: nn.Module,
        visual_dim: int = 1024,
        text_dim: int = 4096,
        llm_dim: int = 4096,
        num_heads: int = 8,
        grid_height: int = 24,
        grid_width: int = 24,
        enable_text_conditioned_vit: bool = True,
        enable_spatial_rope: bool = True,
        enable_spatial_cross_attention: bool = True,
        injection_layers: Optional[list] = None,
    ):
        """
        Args:
            vit_encoder: 预训练 ViT 视觉编码器
            llm_decoder: 预训练 LLM 解码器
            text_encoder: 文本编码器（可以是 LLM 的前几层）
            visual_dim: ViT 输出维度
            text_dim: 文本编码器输出维度
            llm_dim: LLM 隐藏层维度
            num_heads: attention head 数量
            grid_height: 默认 patch grid 高度
            grid_width: 默认 patch grid 宽度
            enable_text_conditioned_vit: 是否启用方案 A
            enable_spatial_rope: 是否启用方案 B
            enable_spatial_cross_attention: 是否启用方案 C
            injection_layers: Text-Conditioned ViT 的注入层索引
        """
        super().__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.llm_dim = llm_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.enable_text_conditioned_vit = enable_text_conditioned_vit
        self.enable_spatial_rope = enable_spatial_rope
        self.enable_spatial_cross_attention = enable_spatial_cross_attention

        # 文本编码器
        self.text_encoder = text_encoder

        # 方案 A：Text-Conditioned ViT
        if enable_text_conditioned_vit:
            self.visual_encoder = TextConditionedViT(
                vit_encoder=vit_encoder,
                text_dim=text_dim,
                injection_layers=injection_layers,
                num_heads=num_heads,
            )
        else:
            self.visual_encoder = vit_encoder

        # 视觉特征投影到 LLM 空间
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # 文本特征投影到 LLM 空间（当 text_dim != llm_dim 时需要）
        if text_dim != llm_dim:
            self.text_projector = nn.Linear(text_dim, llm_dim)
        else:
            self.text_projector = nn.Identity()

        # 方案 B：Hybrid Position Embedding
        # 需要通过 patch_llm_with_spatial_rope() 注入到 LLM 的 attention 层中
        if enable_spatial_rope:
            head_dim = llm_dim // num_heads
            self.hybrid_position_embedding = HybridPositionEmbedding(
                head_dim=head_dim,
                max_seq_len=4096,
                max_grid_size=max(grid_height, grid_width) + 1,
            )
            # 记录原始 rotary embedding 以便恢复
            self._original_rotary_emb_forward = None

        # 方案 C：Spatial-Aware Cross Attention
        if enable_spatial_cross_attention:
            self.spatial_cross_attention = SpatialAwareCrossAttention(
                text_dim=llm_dim,
                visual_dim=llm_dim,
                num_heads=num_heads,
            )

        # LLM 解码器
        self.llm_decoder = llm_decoder

    def patch_llm_with_spatial_rope(self) -> None:
        """将 LLM 内部的 RoPE 替换为 HybridPositionEmbedding。

        通过 monkey-patch 的方式修改 LLM 的 rotary embedding。
        支持 LLaMA 系列模型（包括 LLaMA 3.x）。

        调用时机：在模型初始化完成后、forward 之前调用一次。
        """
        if not self.enable_spatial_rope:
            return

        llm = self.llm_decoder
        hybrid_pe = self.hybrid_position_embedding

        # 查找 LLM 中的 rotary embedding 模块
        rotary_emb = self._find_rotary_emb(llm)
        if rotary_emb is None:
            raise ValueError(
                "Cannot find rotary embedding in LLM decoder. "
                "Supported architectures: LLaMA (model.rotary_emb or model.layers[0].self_attn.rotary_emb)"
            )

        # 保存原始 forward 以便恢复
        self._original_rotary_emb_forward = rotary_emb.forward

        # 创建包装函数：在视觉 token 区间使用 2D RoPE，文本区间使用 1D RoPE
        # 注意：这里只替换 cos/sin 的生成逻辑，实际的旋转操作仍由 LLM attention 层执行
        original_forward = rotary_emb.forward

        def patched_forward(
            x: torch.Tensor,
            position_ids: Optional[torch.Tensor] = None,
            seq_len: Optional[int] = None,
            _hybrid_pe: HybridPositionEmbedding = hybrid_pe,
            _original_forward=original_forward,
            _num_visual_tokens: int = self.grid_height * self.grid_width,
            _grid_h: int = self.grid_height,
            _grid_w: int = self.grid_width,
            **kwargs,
        ):
            # 如果序列长度小于等于视觉 token 数量，说明可能是纯文本推理，使用原始 RoPE
            actual_seq_len = seq_len if seq_len is not None else x.shape[-2]
            if actual_seq_len <= _num_visual_tokens:
                return _original_forward(x, position_ids=position_ids, seq_len=seq_len, **kwargs)

            # 生成 2D cos/sin 用于视觉 token
            visual_cos = _hybrid_pe.spatial_rope.cos_cache[:_num_visual_tokens]
            visual_sin = _hybrid_pe.spatial_rope.sin_cache[:_num_visual_tokens]

            # 生成 1D cos/sin 用于文本 token
            num_text = actual_seq_len - _num_visual_tokens
            text_cos = _hybrid_pe.text_cos_cache[:num_text]
            text_sin = _hybrid_pe.text_sin_cache[:num_text]

            # 拼接：[visual_2d_rope | text_1d_rope]
            combined_cos = torch.cat([visual_cos, text_cos], dim=0)
            combined_sin = torch.cat([visual_sin, text_sin], dim=0)

            return combined_cos, combined_sin

        rotary_emb.forward = patched_forward

    def restore_llm_rope(self) -> None:
        """恢复 LLM 的原始 RoPE，撤销 patch_llm_with_spatial_rope 的修改。"""
        if self._original_rotary_emb_forward is not None:
            rotary_emb = self._find_rotary_emb(self.llm_decoder)
            if rotary_emb is not None:
                rotary_emb.forward = self._original_rotary_emb_forward
            self._original_rotary_emb_forward = None

    @staticmethod
    def _find_rotary_emb(llm: nn.Module):
        """在 LLM 中查找 rotary embedding 模块。"""
        # LLaMA 3.x 风格：model.rotary_emb
        if hasattr(llm, "model") and hasattr(llm.model, "rotary_emb"):
            return llm.model.rotary_emb
        # LLaMA 2.x 风格：每层 self_attn 中有 rotary_emb
        if hasattr(llm, "model") and hasattr(llm.model, "layers"):
            first_layer = llm.model.layers[0]
            if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "rotary_emb"):
                return first_layer.self_attn.rotary_emb
        # 直接在顶层查找
        if hasattr(llm, "rotary_emb"):
            return llm.rotary_emb
        return None

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """编码文本输入。

        支持多种 text_encoder 输出格式：
        - HuggingFace 模型输出（有 last_hidden_state 属性）
        - HuggingFace CausalLM 输出（有 hidden_states 属性，需要 output_hidden_states=True）
        - 直接返回 Tensor 的自定义编码器
        """
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # 优先使用 last_hidden_state（Encoder 模型如 BERT）
        if hasattr(text_outputs, "last_hidden_state"):
            return text_outputs.last_hidden_state

        # CausalLM 模型（如 LLaMA）：使用最后一层 hidden_states
        if hasattr(text_outputs, "hidden_states") and text_outputs.hidden_states is not None:
            return text_outputs.hidden_states[-1]

        # 直接返回 Tensor 的情况
        if isinstance(text_outputs, torch.Tensor):
            return text_outputs

        raise ValueError(
            f"Unsupported text encoder output type: {type(text_outputs)}. "
            "Expected output with 'last_hidden_state', 'hidden_states', or a raw Tensor."
        )

    def encode_visual(
        self,
        pixel_values: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """编码视觉输入，可选地使用文本引导。"""
        if self.enable_text_conditioned_vit and text_tokens is not None:
            visual_features = self.visual_encoder(
                pixel_values=pixel_values,
                text_tokens=text_tokens,
                text_padding_mask=text_padding_mask,
            )
        else:
            visual_features = self.visual_encoder(pixel_values)

        # 投影到 LLM 空间
        projected_visual = self.visual_projector(visual_features)
        return projected_visual

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        grid_height: Optional[int] = None,
        grid_width: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            pixel_values: (batch, channels, height, width) 输入图像
            input_ids: (batch, seq_len) 文本 token ids
            attention_mask: (batch, seq_len) 文本 attention mask
            labels: (batch, seq_len) 训练标签
            grid_height: patch grid 高度（可选，默认使用初始化值）
            grid_width: patch grid 宽度（可选）

        Returns:
            dict with keys: logits, loss (if labels provided), attention_weights
        """
        grid_height = grid_height or self.grid_height
        grid_width = grid_width or self.grid_width

        # Step 1: 编码文本
        text_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        text_hidden_states = self.encode_text(input_ids, attention_mask)

        # Step 2: 编码视觉（可选文本引导）
        visual_features = self.encode_visual(
            pixel_values=pixel_values,
            text_tokens=text_hidden_states,
            text_padding_mask=text_padding_mask,
        )

        # Step 3: 将文本特征投影到 LLM 空间（处理 text_dim != llm_dim 的情况）
        text_in_llm_space = self.text_projector(text_hidden_states)

        # Step 4: 空间感知的 Cross Attention（方案 C）
        if self.enable_spatial_cross_attention:
            aligned_features, cross_attention_weights = self.spatial_cross_attention(
                text_hidden_states=text_in_llm_space,
                visual_features=visual_features,
                grid_height=grid_height,
                grid_width=grid_width,
                text_padding_mask=text_padding_mask,
            )
        else:
            aligned_features = text_in_llm_space
            cross_attention_weights = None

        # Step 5: 拼接视觉和文本 token 送入 LLM
        # 排列顺序：[visual_tokens | text_tokens]
        # 方案 B 的 HybridPositionEmbedding 依赖此排列顺序，
        # 在 patch_llm_with_spatial_rope() 中已将 LLM 的 RoPE 替换为
        # 对前 num_visual_tokens 个位置使用 2D RoPE，后续位置使用 1D RoPE
        combined_input = torch.cat([visual_features, aligned_features], dim=1)

        # 构建 combined attention mask
        batch_size = pixel_values.shape[0]
        num_visual_tokens = visual_features.shape[1]
        visual_attention_mask = torch.ones(
            batch_size, num_visual_tokens, dtype=torch.long, device=pixel_values.device
        )
        if attention_mask is not None:
            combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
        else:
            text_attention_mask = torch.ones(
                batch_size, aligned_features.shape[1], dtype=torch.long, device=pixel_values.device
            )
            combined_attention_mask = torch.cat([visual_attention_mask, text_attention_mask], dim=1)

        # Step 6: LLM 解码
        outputs = self.llm_decoder(
            inputs_embeds=combined_input,
            attention_mask=combined_attention_mask,
            labels=labels,
        )

        result = {
            "logits": outputs.logits if hasattr(outputs, "logits") else outputs,
            "visual_features": visual_features,
            "cross_attention_weights": cross_attention_weights,
        }

        if labels is not None and hasattr(outputs, "loss"):
            result["loss"] = outputs.loss

        return result