"""
Spatial VLM：基于 Qwen3.5-4B 的空间增强 VLM。

Qwen3.5-4B 是原生多模态模型，自带 ViT + LLM + 投影层。
本模块在 Qwen3.5 的基础上注入两个空间增强模块：

- 方案 A：Text-Conditioned ViT (KV Injection)
  在 Qwen3.5 ViT 的每层 attention 中注入文本 context KV，
  让视觉编码被文本意图引导（"先知道要看什么，再去看"）。

- 方案 C：Latent Reasoning Loop
  在 ViT 输出和 LLM 之间插入隐空间迭代推理模块，
  模拟人类"逐个看、然后串联"的空间推理过程。
  reasoning tokens 作为额外的 prefix tokens 拼接到 LLM 输入前面。

方案 B（2D RoPE）不需要额外实现，因为 Qwen3.5 已自带 mrope。

架构：
  Qwen3.5-4B (frozen/trainable)
    ├── ViT (depth=24, hidden=1024, heads=16)
    │   └── [方案 A] KV Injection: 每层注入文本 context
    ├── PatchMerger (1024*4 → 2560)
    ├── [方案 C] Latent Reasoning Loop (visual=1024, text=2560 → 2560)
    └── LLM (hidden=2560, heads=16, layers=32, hybrid attention)

Qwen3.5 内部结构（关键路径）：
  Qwen3_5ForConditionalGeneration
    ├── model (Qwen3_5Model)
    │   ├── visual (Qwen3_5VisionModel)
    │   │   ├── blocks[0..23] (Qwen3_5VisionBlock)
    │   │   └── merger (Qwen3_5VisionPatchMerger)
    │   └── language_model (Qwen3_5TextModel)
    │       ├── embed_tokens
    │       ├── layers[0..31] (Qwen3_5DecoderLayer)
    │       └── norm
    └── lm_head
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from src.model.text_conditioned_vit.text_conditioned_vit_kv import TextConditionedViTKV
from src.model.latent_reasoning.latent_reasoning_loop import LatentReasoningLoop


class SpatialVLM(nn.Module):
    """基于 Qwen3.5-4B 的空间增强 Vision-Language Model。

    包装 Qwen3.5-4B 整体模型，注入方案 A 和方案 C。
    训练时可以选择冻结 ViT/LLM，只训练新增模块。

    方案 C 的集成策略：
    不使用 hook（因为 Qwen3.5 的 masked_scatter 要求 image_embeds 长度精确匹配
    input_ids 中的 placeholder token 数量）。而是手动拆解 forward 流程：
    1. 手动调用 ViT 获取 image_embeds
    2. 用 ViT 的原始输出（merger 之前）做 Latent Reasoning
    3. 将 reasoning tokens 作为额外 prefix 拼接到 inputs_embeds 前面
    4. 调整 attention_mask、position_ids、labels 的长度
    5. 调用 LLM forward
    """

    def __init__(
        self,
        base_model: nn.Module,
        enable_text_conditioned_vit: bool = True,
        enable_latent_reasoning_loop: bool = True,
        # 方案 A 参数
        text_dim: int = 2560,
        context_dim: int = 256,
        injection_layers: Optional[List[int]] = None,
        vit_num_heads: int = 16,
        # 方案 C 参数
        visual_dim: int = 1024,
        latent_dim: int = 512,
        num_latent_tokens: int = 8,
        num_iterations: int = 4,
        reasoning_num_heads: int = 8,
        reasoning_output_dim: int = 2560,
        share_reasoning_weights: bool = True,
    ):
        """
        Args:
            base_model: Qwen3_5ForConditionalGeneration 实例
            enable_text_conditioned_vit: 是否启用方案 A
            enable_latent_reasoning_loop: 是否启用方案 C
            text_dim: 文本/LLM hidden_size (2560)
            context_dim: KV injection 的 context 压缩维度
            injection_layers: 注入哪些 ViT 层，None = 所有 24 层
            vit_num_heads: ViT attention heads (16)
            visual_dim: ViT hidden_size (1024)
            latent_dim: Reasoning Loop 隐空间维度
            num_latent_tokens: Reasoning Loop latent tokens 数量
            num_iterations: Reasoning Loop 迭代轮数
            reasoning_num_heads: Reasoning Loop attention heads
            reasoning_output_dim: Reasoning Loop 输出维度 (= LLM dim = 2560)
            share_reasoning_weights: Reasoning Loop 是否共享权重
        """
        super().__init__()

        self.base_model = base_model
        self.enable_text_conditioned_vit = enable_text_conditioned_vit
        self.enable_latent_reasoning_loop = enable_latent_reasoning_loop
        self.num_latent_tokens = num_latent_tokens

        # 方案 A：Text-Conditioned ViT (KV Injection)
        if enable_text_conditioned_vit:
            vision_model = self._get_vision_model()
            self.text_conditioned_vit = TextConditionedViTKV(
                vision_model=vision_model,
                text_dim=text_dim,
                context_dim=context_dim,
                injection_layers=injection_layers,
                num_heads=vit_num_heads,
            )

        # 方案 C：Latent Reasoning Loop
        if enable_latent_reasoning_loop:
            self.latent_reasoning_loop = LatentReasoningLoop(
                visual_dim=visual_dim,
                text_dim=text_dim,
                latent_dim=latent_dim,
                num_latent_tokens=num_latent_tokens,
                num_iterations=num_iterations,
                num_heads=reasoning_num_heads,
                output_dim=reasoning_output_dim,
                max_grid_size=32,
                share_weights=share_reasoning_weights,
            )

    def _get_vision_model(self) -> nn.Module:
        """从 Qwen3.5 base_model 中提取 VisionModel。

        Qwen3_5ForConditionalGeneration 结构:
          base_model.model (Qwen3_5Model)
            ├── visual (Qwen3_5VisionModel)
            └── language_model (Qwen3_5TextModel)
        """
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "visual"):
            return self.base_model.model.visual
        raise ValueError(
            "Cannot find vision model in base_model. "
            "Expected: base_model.model.visual (Qwen3_5ForConditionalGeneration)"
        )

    def _get_language_model(self) -> nn.Module:
        """从 Qwen3.5 base_model 中提取 LLM (TextModel)。"""
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "language_model"):
            return self.base_model.model.language_model
        raise ValueError(
            "Cannot find language model in base_model. "
            "Expected: base_model.model.language_model"
        )

    def _get_qwen3_5_model(self) -> nn.Module:
        """获取 Qwen3_5Model（包含 visual + language_model）。"""
        if hasattr(self.base_model, "model"):
            return self.base_model.model
        raise ValueError("Cannot find Qwen3_5Model in base_model.")

    def _get_embedding_layer(self) -> nn.Module:
        """获取 LLM 的 embedding 层。"""
        language_model = self._get_language_model()
        if hasattr(language_model, "embed_tokens"):
            return language_model.embed_tokens
        return language_model.get_input_embeddings()

    def freeze_vision(self) -> None:
        """冻结 ViT 参数（方案 A 的新增参数不冻结）。"""
        vision_model = self._get_vision_model()
        for param in vision_model.parameters():
            param.requires_grad = False

    def freeze_llm(self) -> None:
        """冻结 LLM 参数（包括 lm_head）。"""
        language_model = self._get_language_model()
        for param in language_model.parameters():
            param.requires_grad = False
        if hasattr(self.base_model, "lm_head"):
            for param in self.base_model.lm_head.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """解冻所有参数。"""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params_info(self) -> Dict[str, int]:
        """获取可训练参数统计。"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_ratio": trainable_params / max(total_params, 1),
        }

        if self.enable_text_conditioned_vit:
            info["text_conditioned_vit_params"] = sum(
                p.numel() for p in self.text_conditioned_vit.parameters()
                if p.requires_grad
            )

        if self.enable_latent_reasoning_loop:
            info["latent_reasoning_loop_params"] = sum(
                p.numel() for p in self.latent_reasoning_loop.parameters()
                if p.requires_grad
            )

        return info

    def _extract_text_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从 input_ids 中提取文本 token 的 embeddings。

        返回 padded 的文本特征和 padding mask，用于方案 A 和方案 C。

        Args:
            input_ids: (B, seq_len)

        Returns:
            text_features: (B, max_text_len, hidden_size)
            text_padding_mask: (B, max_text_len) True = padding
            text_mask: (B, seq_len) True = 文本 token（非视觉 placeholder）
        """
        config = self.base_model.config
        special_ids = {
            getattr(config, "image_token_id", 248056),
            getattr(config, "video_token_id", 248057),
            getattr(config, "vision_start_token_id", 248053),
            getattr(config, "vision_end_token_id", 248054),
        }

        # 找到文本 token（非视觉 placeholder）
        text_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for sid in special_ids:
            text_mask = text_mask & (input_ids != sid)

        # 获取 embeddings
        embed_layer = self._get_embedding_layer()
        with torch.no_grad():
            embeddings = embed_layer(input_ids)

        batch_size = input_ids.shape[0]
        max_text_len = text_mask.sum(dim=1).max().item()

        if max_text_len == 0:
            hidden_size = embeddings.shape[-1]
            empty_features = torch.zeros(
                batch_size, 1, hidden_size,
                dtype=embeddings.dtype, device=embeddings.device,
            )
            empty_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=embeddings.device,
            )
            return empty_features, empty_mask, text_mask

        text_features = torch.zeros(
            batch_size, max_text_len, embeddings.shape[-1],
            dtype=embeddings.dtype, device=embeddings.device,
        )
        text_padding_mask = torch.ones(
            batch_size, max_text_len,
            dtype=torch.bool, device=embeddings.device,
        )

        for b in range(batch_size):
            valid_embeds = embeddings[b][text_mask[b]]
            actual_len = valid_embeds.shape[0]
            text_features[b, :actual_len] = valid_embeds
            text_padding_mask[b, :actual_len] = False

        return text_features, text_padding_mask, text_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass，手动拆解 Qwen3.5 的流程以集成方案 A 和方案 C。

        当方案 C 启用且有图像输入时，流程为：
        1. 提取文本 embeddings → 设置方案 A 的 text context
        2. 手动调用 ViT 获取 image_embeds（方案 A 的 KV injection 在此生效）
        3. 用 ViT 原始输出做 Latent Reasoning → reasoning_tokens
        4. 构建 inputs_embeds：将 image_embeds 替换 placeholder + 拼接 reasoning_tokens
        5. 调整 attention_mask / position_ids / labels
        6. 调用 LLM forward

        当方案 C 未启用时，直接代理到 base_model.forward（方案 A 仍通过 monkey-patch 生效）。

        Args:
            input_ids: (B, seq_len) 文本 token ids（包含 image placeholder tokens）
            attention_mask: (B, seq_len) attention mask
            pixel_values: 图像 patch 像素值
            image_grid_thw: (num_images, 3) 每张图的 (temporal, height, width)
            pixel_values_videos: 视频 patch 像素值
            video_grid_thw: (num_videos, 3) 每个视频的 grid
            mm_token_type_ids: (B, seq_len) 多模态 token 类型 ids
            labels: (B, seq_len) 训练标签
            **kwargs: 传递给 base_model 的其他参数
        """
        has_images = pixel_values is not None
        use_manual_forward = self.enable_latent_reasoning_loop and has_images

        # 方案 A：设置 text context（无论是否手动 forward 都需要）
        if self.enable_text_conditioned_vit and has_images and input_ids is not None:
            text_features, text_padding_mask, _ = self._extract_text_embeddings(input_ids)
            self.text_conditioned_vit.set_text_context(
                text_features, text_padding_mask=text_padding_mask
            )

        try:
            if use_manual_forward:
                return self._forward_with_reasoning(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    labels=labels,
                    **kwargs,
                )
            else:
                return self._forward_passthrough(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    labels=labels,
                    **kwargs,
                )
        finally:
            if self.enable_text_conditioned_vit:
                self.text_conditioned_vit.clear_text_context()

    def _forward_passthrough(self, **kwargs) -> Dict[str, Any]:
        """直接代理到 base_model.forward（方案 A 通过 monkey-patch 生效）。"""
        outputs = self.base_model(**kwargs)
        result = {
            "logits": outputs.logits if hasattr(outputs, "logits") else None,
        }
        if hasattr(outputs, "loss") and outputs.loss is not None:
            result["loss"] = outputs.loss
        return result

    def _forward_with_reasoning(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        mm_token_type_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """手动拆解 forward，在 ViT 输出后插入 Latent Reasoning tokens。

        核心思路：reasoning tokens 作为额外的 prefix 拼接到 inputs_embeds 前面，
        这样不会破坏 Qwen3.5 的 image placeholder 替换逻辑。
        """
        qwen_model = self._get_qwen3_5_model()
        batch_size = input_ids.shape[0]

        # Step 1: 获取 text embeddings
        embed_layer = self._get_embedding_layer()
        inputs_embeds = embed_layer(input_ids)

        # Step 2: 调用 ViT 获取 image_embeds（方案 A 的 KV injection 在此生效）
        image_embeds = None
        raw_vit_features = None
        if pixel_values is not None:
            vision_output = qwen_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            # pooler_output 是 merger 后的 image_embeds（list of tensors）
            image_embeds_list = vision_output.pooler_output
            image_embeds = torch.cat(image_embeds_list, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            # last_hidden_state 是 merger 前的原始 ViT 特征
            raw_vit_features = vision_output.last_hidden_state

        # Step 3: 用 masked_scatter 替换 image placeholder tokens
        if image_embeds is not None:
            image_mask, _ = qwen_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Step 4: 处理视频（如果有）
        if pixel_values_videos is not None:
            video_output = qwen_model.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = torch.cat(video_output.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = qwen_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Step 5: Latent Reasoning Loop
        reasoning_tokens = None
        if raw_vit_features is not None:
            reasoning_tokens = self._compute_reasoning_tokens(
                raw_vit_features, image_grid_thw, input_ids
            )
            # reasoning_tokens: (B, K, llm_dim)

        # Step 6: 拼接 reasoning tokens 作为 prefix
        if reasoning_tokens is not None:
            num_reasoning = reasoning_tokens.shape[1]
            inputs_embeds = torch.cat([reasoning_tokens, inputs_embeds], dim=1)

            # 扩展 attention_mask
            if attention_mask is not None:
                reasoning_mask = torch.ones(
                    batch_size, num_reasoning,
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                attention_mask = torch.cat([reasoning_mask, attention_mask], dim=1)

            # 扩展 labels：reasoning tokens 位置设为 -100（不计算 loss）
            if labels is not None:
                ignore_labels = torch.full(
                    (batch_size, num_reasoning), -100,
                    dtype=labels.dtype, device=labels.device,
                )
                labels = torch.cat([ignore_labels, labels], dim=1)

            # 扩展 mm_token_type_ids：reasoning tokens 标记为 text type (0)
            if mm_token_type_ids is not None:
                reasoning_type = torch.zeros(
                    batch_size, num_reasoning,
                    dtype=mm_token_type_ids.dtype, device=mm_token_type_ids.device,
                )
                mm_token_type_ids = torch.cat([reasoning_type, mm_token_type_ids], dim=1)

        # Step 7: 计算 position_ids
        # compute_3d_position_ids 需要 input_ids + mm_token_type_ids 来计算 mrope。
        # 由于我们在前面 prepend 了 reasoning tokens，需要构建扩展后的 input_ids
        # 用于 position_ids 计算（reasoning prefix 用 pad_token_id 填充）。
        if reasoning_tokens is not None and mm_token_type_ids is not None:
            pad_token_id = getattr(
                self.base_model.config, "pad_token_id",
                getattr(self.base_model.config.text_config, "pad_token_id", 0),
            ) or 0
            reasoning_input_ids = torch.full(
                (batch_size, num_reasoning), pad_token_id,
                dtype=input_ids.dtype, device=input_ids.device,
            )
            extended_input_ids = torch.cat([reasoning_input_ids, input_ids], dim=1)
        else:
            extended_input_ids = input_ids

        position_ids = qwen_model.compute_3d_position_ids(
            input_ids=extended_input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=kwargs.get("past_key_values"),
            mm_token_type_ids=mm_token_type_ids,
        )

        # Step 8: 调用 LLM forward
        language_model = self._get_language_model()
        lm_outputs = language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = lm_outputs.last_hidden_state
        logits = self.base_model.lm_head(hidden_states)

        # Step 9: 计算 loss
        loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return {"loss": loss, "logits": logits}

    def _compute_reasoning_tokens(
        self,
        raw_vit_features: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """用 ViT 原始特征和文本 embeddings 做 Latent Reasoning。

        Args:
            raw_vit_features: (total_patches, visual_dim) ViT blocks 输出（packed 2D）
            image_grid_thw: (num_images, 3) 每张图的 (T, H, W)
            input_ids: (B, seq_len) 用于提取文本 embeddings

        Returns:
            reasoning_tokens: (B, K, output_dim)
        """
        batch_size = input_ids.shape[0]

        # 提取文本 embeddings 用于 reasoning
        text_features, text_padding_mask, _ = self._extract_text_embeddings(input_ids)

        # 将 packed ViT features unpack 成 per-image tensors
        # image_grid_thw: (num_images, 3) → 每张图的 patch 数量 = T * H * W
        patches_per_image = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        image_features_list = torch.split(raw_vit_features, patches_per_image, dim=0)

        # 对每张图独立做 reasoning，然后 stack 成 batch
        # 映射策略：第 i 张图对应第 min(i, B-1) 个 batch item 的文本
        # 当 num_images == batch_size 时为一一对应；否则 clamp 到有效范围
        num_images = len(image_features_list)
        all_reasoning = []

        for img_idx in range(num_images):
            img_features = image_features_list[img_idx].unsqueeze(0)  # (1, N, visual_dim)
            grid_t, grid_h, grid_w = image_grid_thw[img_idx].tolist()

            # 确定 batch 中对应的文本特征
            # 简单映射：第 i 张图对应第 i 个 batch item
            batch_idx = min(img_idx, batch_size - 1)
            img_text = text_features[batch_idx:batch_idx + 1]  # (1, T, text_dim)
            img_text_mask = text_padding_mask[batch_idx:batch_idx + 1] if text_padding_mask is not None else None

            reasoning = self.latent_reasoning_loop(
                patch_features=img_features,
                text_hidden_states=img_text,
                text_padding_mask=img_text_mask,
                grid_height=grid_h,
                grid_width=grid_w,
            )  # (1, K, output_dim)
            all_reasoning.append(reasoning)

        # Stack 成 batch
        if num_images == batch_size:
            reasoning_tokens = torch.cat(all_reasoning, dim=0)  # (B, K, output_dim)
        elif num_images < batch_size:
            # 图片数少于 batch size：用最后一张图的 reasoning 填充
            reasoning_tokens = torch.cat(all_reasoning, dim=0)
            if reasoning_tokens.shape[0] < batch_size:
                padding = reasoning_tokens[-1:].expand(
                    batch_size - reasoning_tokens.shape[0], -1, -1
                )
                reasoning_tokens = torch.cat([reasoning_tokens, padding], dim=0)
        else:
            # 图片数多于 batch size：截断
            reasoning_tokens = torch.cat(all_reasoning[:batch_size], dim=0)

        return reasoning_tokens

    def generate(self, **kwargs):
        """生成接口，支持方案 A 和方案 C。"""
        has_images = kwargs.get("pixel_values") is not None
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")

        # 方案 A：设置 text context
        if self.enable_text_conditioned_vit and has_images and input_ids is not None:
            text_features, text_padding_mask, _ = self._extract_text_embeddings(input_ids)
            self.text_conditioned_vit.set_text_context(
                text_features, text_padding_mask=text_padding_mask
            )

        try:
            if self.enable_latent_reasoning_loop and has_images:
                # 方案 C：需要预处理 reasoning tokens
                return self._generate_with_reasoning(**kwargs)
            else:
                return self.base_model.generate(**kwargs)
        finally:
            if self.enable_text_conditioned_vit:
                self.text_conditioned_vit.clear_text_context()

    def _generate_with_reasoning(self, **kwargs) -> Any:
        """带 Latent Reasoning 的生成。

        先计算 reasoning tokens，拼接到 inputs_embeds 前面，
        然后用 inputs_embeds 模式调用 generate。
        """
        input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs.pop("attention_mask", None)
        pixel_values = kwargs.pop("pixel_values")
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)

        qwen_model = self._get_qwen3_5_model()
        batch_size = input_ids.shape[0]

        # 获取 embeddings 和 image features
        embed_layer = self._get_embedding_layer()
        inputs_embeds = embed_layer(input_ids)

        vision_output = qwen_model.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = torch.cat(vision_output.pooler_output, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        raw_vit_features = vision_output.last_hidden_state

        # 替换 image placeholders
        image_mask, _ = qwen_model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Reasoning tokens
        reasoning_tokens = self._compute_reasoning_tokens(
            raw_vit_features, image_grid_thw, input_ids
        )
        num_reasoning = reasoning_tokens.shape[1]
        inputs_embeds = torch.cat([reasoning_tokens, inputs_embeds], dim=1)

        if attention_mask is not None:
            reasoning_mask = torch.ones(
                batch_size, num_reasoning,
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            attention_mask = torch.cat([reasoning_mask, attention_mask], dim=1)

        if mm_token_type_ids is not None:
            reasoning_type = torch.zeros(
                batch_size, num_reasoning,
                dtype=mm_token_type_ids.dtype, device=mm_token_type_ids.device,
            )
            mm_token_type_ids = torch.cat([reasoning_type, mm_token_type_ids], dim=1)

        # 调用 generate（用 inputs_embeds 模式，不传 input_ids 和 pixel_values）
        # 不传 image_grid_thw，因为图像已经被嵌入到 inputs_embeds 中，
        # generate 内部的后续 forward 调用不需要再处理图像。
        return self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )