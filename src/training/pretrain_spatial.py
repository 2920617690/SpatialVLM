"""
Text-Conditioned Latent Prediction 预训练

核心创新：将 V-JEPA 的隐空间预测与文本条件化 KV Injection 结合。
- Target Encoder (EMA): 原始 ViT 看完整图像 → target features
- Context Encoder: TextConditionedViTKV 看部分图像 + 文本 KV → context features
- Predictor: 轻量 transformer 预测被遮挡区域的隐空间特征
- 文本条件化空间掩码：根据文本空间语义决定遮哪里，迫使模型学会"带着问题去看"
"""

import copy
import math
import random
import re
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. SpatialMaskGenerator — 文本条件化的空间掩码生成
# ============================================================

class SpatialMaskGenerator:
    """根据文本空间语义生成 patch-level 掩码。

    三种策略按概率混合：
    - directional: 文本含空间方向词 → 遮对应半区
    - block: 文本有 bbox → 遮 bbox 覆盖的 patches
    - random: 随机连续 block，作为正则化
    """

    # 空间关键词 → 要遮挡的方向
    SPATIAL_KEYWORDS: Dict[str, str] = {
        # 中文
        "左边": "left", "左侧": "left", "左方": "left", "左面": "left",
        "右边": "right", "右侧": "right", "右方": "right", "右面": "right",
        "上边": "top", "上方": "top", "上面": "top", "顶部": "top",
        "下边": "bottom", "下方": "bottom", "下面": "bottom", "底部": "bottom",
        # English
        "left": "left", "on the left": "left", "to the left": "left",
        "right": "right", "on the right": "right", "to the right": "right",
        "top": "top", "above": "top", "upper": "top", "on top": "top",
        "bottom": "bottom", "below": "bottom", "lower": "bottom", "beneath": "bottom",
    }

    # 反转映射（水平翻转增强时交换左右）
    DIRECTION_FLIP: Dict[str, str] = {
        "left": "right", "right": "left", "top": "top", "bottom": "bottom",
    }

    def __init__(
        self,
        mask_ratio: float = 0.5,
        min_mask_ratio: float = 0.3,
        max_mask_ratio: float = 0.7,
        directional_prob: float = 0.5,
        block_prob: float = 0.25,
        random_prob: float = 0.25,
    ):
        self.mask_ratio = mask_ratio
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.directional_prob = directional_prob
        self.block_prob = block_prob
        self.random_prob = random_prob

    def detect_spatial_direction(self, text: str) -> Optional[str]:
        """从文本中检测空间方向词，返回方向或 None。"""
        text_lower = text.lower()
        for keyword, direction in self.SPATIAL_KEYWORDS.items():
            if keyword in text_lower:
                return direction
        return None

    def generate(
        self,
        texts: List[str],
        bboxes: List[Optional[Tuple[float, float, float, float]]],
        grid_h: int = 27,
        grid_w: int = 27,
    ) -> torch.Tensor:
        """生成 batch 的掩码。

        Args:
            texts: 每个样本的原始文本
            bboxes: 每个样本的归一化 bbox [x1,y1,x2,y2]，无则 None
            grid_h, grid_w: patch grid 尺寸

        Returns:
            mask: (B, grid_h * grid_w) bool tensor, True = 被遮挡
        """
        B = len(texts)
        N = grid_h * grid_w
        masks = torch.zeros(B, N, dtype=torch.bool)

        for i in range(B):
            direction = self.detect_spatial_direction(texts[i])
            has_bbox = bboxes[i] is not None

            # 选择策略
            r = random.random()
            if direction is not None and r < self.directional_prob:
                masks[i] = self._directional_mask(direction, grid_h, grid_w)
            elif has_bbox and r < self.directional_prob + self.block_prob:
                masks[i] = self._block_mask(bboxes[i], grid_h, grid_w)
            else:
                masks[i] = self._random_block_mask(grid_h, grid_w)

        return masks

    def _directional_mask(
        self, direction: str, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """遮挡指定方向的半区。"""
        mask = torch.zeros(grid_h, grid_w, dtype=torch.bool)
        if direction == "left":
            mask[:, : grid_w // 2] = True
        elif direction == "right":
            mask[:, grid_w // 2 :] = True
        elif direction == "top":
            mask[: grid_h // 2, :] = True
        elif direction == "bottom":
            mask[grid_h // 2 :, :] = True
        return mask.flatten()

    def _block_mask(
        self,
        bbox: Tuple[float, float, float, float],
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """遮挡 bbox 覆盖的 patch 区域。"""
        x1, y1, x2, y2 = bbox
        # 归一化坐标 → patch grid 坐标
        col_start = int(x1 * grid_w)
        col_end = min(int(math.ceil(x2 * grid_w)), grid_w)
        row_start = int(y1 * grid_h)
        row_end = min(int(math.ceil(y2 * grid_h)), grid_h)

        mask = torch.zeros(grid_h, grid_w, dtype=torch.bool)
        mask[row_start:row_end, col_start:col_end] = True

        # 确保 mask ratio 在合理范围
        ratio = mask.float().mean().item()
        if ratio < self.min_mask_ratio:
            # bbox 太小，扩展到周围
            mask = self._expand_mask(mask, grid_h, grid_w, self.min_mask_ratio)
        elif ratio > self.max_mask_ratio:
            # bbox 太大，随机保留一部分
            mask = self._shrink_mask(mask, grid_h, grid_w, self.max_mask_ratio)

        return mask.flatten()

    def _random_block_mask(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """随机连续 block 掩码。"""
        target_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        target_patches = int(target_ratio * grid_h * grid_w)

        # 随机选择 block 的大小和位置
        block_h = random.randint(grid_h // 4, grid_h * 3 // 4)
        block_w = max(1, target_patches // block_h)
        block_w = min(block_w, grid_w)

        top = random.randint(0, grid_h - block_h)
        left = random.randint(0, max(1, grid_w - block_w))

        mask = torch.zeros(grid_h, grid_w, dtype=torch.bool)
        mask[top : top + block_h, left : left + block_w] = True
        return mask.flatten()

    def _expand_mask(
        self, mask: torch.Tensor, grid_h: int, grid_w: int, target_ratio: float
    ) -> torch.Tensor:
        """扩展 mask 到目标比例。"""
        target_count = int(target_ratio * grid_h * grid_w)
        current = mask.clone()
        # 膨胀操作：向四周扩展
        while current.sum() < target_count:
            padded = F.pad(current.float().unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="constant", value=0)
            dilated = F.max_pool2d(padded, kernel_size=3, stride=1, padding=0)
            current = dilated.squeeze().bool()[:grid_h, :grid_w]
        return current

    def _shrink_mask(
        self, mask: torch.Tensor, grid_h: int, grid_w: int, target_ratio: float
    ) -> torch.Tensor:
        """收缩 mask 到目标比例。"""
        target_count = int(target_ratio * grid_h * grid_w)
        indices = mask.flatten().nonzero(as_tuple=True)[0]
        keep = indices[torch.randperm(len(indices))[:target_count]]
        new_mask = torch.zeros(grid_h * grid_w, dtype=torch.bool)
        new_mask[keep] = True
        return new_mask.view(grid_h, grid_w)


# ============================================================
# 2. LatentPredictor — 轻量 Transformer 预测器
# ============================================================

class PredictorBlock(nn.Module):
    """Predictor 的一个 Transformer block: self-attn + cross-attn(text) + FFN。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention to text context
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_text = nn.LayerNorm(dim)

        # FFN
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        text_context: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim)
            text_context: (B, T, dim)
            text_padding_mask: (B, T) True = padding
        Returns:
            x: (B, N, dim)
        """
        # Self-attention
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, normed)[0]

        # Cross-attention to text
        normed = self.norm2(x)
        text_normed = self.norm_text(text_context)
        x = x + self.cross_attn(
            normed, text_normed, text_normed,
            key_padding_mask=text_padding_mask,
        )[0]

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class LatentPredictor(nn.Module):
    """轻量 Transformer 预测器：预测被遮挡 patches 的隐空间特征。

    输入：context features (可见位置) + mask tokens (遮挡位置) + 文本 cross-attn
    输出：遮挡位置的 predicted features
    """

    def __init__(
        self,
        visual_dim: int = 1152,
        text_context_dim: int = 256,
        predictor_dim: int = 384,
        num_layers: int = 3,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_patches: int = 729,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.predictor_dim = predictor_dim

        # 可学习的 mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, visual_dim) * 0.02)

        # 2D positional embedding for patch grid
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, visual_dim) * 0.02)

        # 输入/输出投影（瓶颈设计）
        self.input_proj = nn.Linear(visual_dim, predictor_dim)
        self.text_proj = nn.Linear(text_context_dim, predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, visual_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PredictorBlock(predictor_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

    def forward(
        self,
        context_features: torch.Tensor,
        mask: torch.Tensor,
        text_context: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context_features: (B, N, visual_dim) context encoder 输出
            mask: (B, N) bool, True = 遮挡位置
            text_context: (B, T, text_context_dim) 文本 context
            text_padding_mask: (B, T) bool

        Returns:
            predictions: (B, N_masked, visual_dim) 遮挡位置的预测特征
        """
        B, N, D = context_features.shape

        # 组装输入：可见位置用 context features，遮挡位置用 mask token
        tokens = context_features.clone()
        tokens[mask] = self.mask_token.expand(B, N, -1)[mask]

        # 加 positional embedding
        tokens = tokens + self.pos_embed[:, :N, :]

        # 投影到 predictor 维度
        tokens = self.input_proj(tokens)
        text_proj = self.text_proj(text_context)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, text_proj, text_padding_mask)

        tokens = self.norm(tokens)

        # 投影回 visual 维度
        tokens = self.output_proj(tokens)

        # 只返回遮挡位置的预测
        predictions = tokens[mask].view(B, -1, D)
        return predictions


# ============================================================
# 3. TextConditionedPretrainer — 主预训练模块
# ============================================================

class TextConditionedPretrainer(nn.Module):
    """Text-Conditioned Latent Prediction 预训练器。

    整合 target encoder (EMA) + context encoder (KV injection) + predictor。
    训练 TextContextEncoder 和 KVInjectionHeads 学会利用文本空间语义来
    引导 ViT 理解图像的空间结构。

    预训练后，只保留 TextContextEncoder + KVInjectionHeads 迁移到 SpatialVLM。
    LatentPredictor 和 target_encoder 丢弃。
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        vit_encoder: nn.Module,
        text_context_dim: int = 256,
        visual_dim: int = 1152,
        predictor_dim: int = 384,
        predictor_layers: int = 3,
        predictor_heads: int = 6,
        ema_decay: float = 0.996,
        ema_decay_end: float = 1.0,
        loss_type: str = "smooth_l1",
        feature_norm: bool = True,
        mask_ratio: float = 0.5,
        directional_prob: float = 0.5,
        block_prob: float = 0.25,
        random_prob: float = 0.25,
    ):
        """
        Args:
            context_encoder: TextConditionedViTKV 实例（trainable）
            vit_encoder: 原始 SigLIP ViT（用于初始化 target encoder）
            text_context_dim: TextContextEncoder 输出维度
            visual_dim: ViT 隐藏维度
            predictor_dim: Predictor 瓶颈维度
            predictor_layers: Predictor transformer 层数
            predictor_heads: Predictor attention head 数
            ema_decay: EMA 初始衰减率
            ema_decay_end: EMA 最终衰减率
            loss_type: "smooth_l1" 或 "cosine"
            feature_norm: 是否 L2 归一化特征再计算 loss
            mask_ratio: 目标掩码比例
            directional_prob: 方向掩码概率
            block_prob: Block 掩码概率
            random_prob: 随机掩码概率
        """
        super().__init__()

        # Context encoder (trainable KV injection modules)
        self.context_encoder = context_encoder

        # Target encoder (EMA, frozen)
        self.target_encoder = copy.deepcopy(vit_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = LatentPredictor(
            visual_dim=visual_dim,
            text_context_dim=text_context_dim,
            predictor_dim=predictor_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
        )

        # Mask generator
        self.mask_generator = SpatialMaskGenerator(
            mask_ratio=mask_ratio,
            directional_prob=directional_prob,
            block_prob=block_prob,
            random_prob=random_prob,
        )

        # EMA 参数
        self.ema_decay = ema_decay
        self.ema_decay_end = ema_decay_end

        # Loss 配置
        self.loss_type = loss_type
        self.feature_norm = feature_norm

        # 可学习的 mask embedding（替换被遮挡的 patch embeddings）
        self.mask_embed = nn.Parameter(torch.randn(1, 1, visual_dim) * 0.02)

    @torch.no_grad()
    def update_target_encoder(self, progress: float = 0.0) -> None:
        """EMA 更新 target encoder。

        Args:
            progress: 训练进度 [0, 1]，用于线性调度 decay
        """
        decay = self.ema_decay + (self.ema_decay_end - self.ema_decay) * progress
        # 只更新 ViT backbone 权重（context_encoder 内部的 vit_encoder）
        for param_t, param_c in zip(
            self.target_encoder.parameters(),
            self.context_encoder.vit_encoder.parameters(),
        ):
            param_t.data.mul_(decay).add_(param_c.data, alpha=1 - decay)

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """计算预测与目标之间的 loss。

        Args:
            predictions: (B, N_masked, D)
            targets: (B, N_masked, D)
        """
        if self.feature_norm:
            predictions = F.normalize(predictions, dim=-1)
            targets = F.normalize(targets, dim=-1)

        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(predictions, targets)
        elif self.loss_type == "cosine":
            return (1 - F.cosine_similarity(predictions, targets, dim=-1)).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: torch.Tensor,
        texts: List[str],
        bboxes: List[Optional[Tuple[float, float, float, float]]],
        text_padding_mask: Optional[torch.Tensor] = None,
        grid_h: int = 27,
        grid_w: int = 27,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W)
            text_tokens: (B, T, text_dim) 文本编码器输出
            texts: List[str] 原始文本（用于掩码生成）
            bboxes: List[Optional[Tuple]] 每个样本的 bbox
            text_padding_mask: (B, T) bool
            grid_h, grid_w: patch grid 尺寸

        Returns:
            dict with: loss, predictions, targets, mask
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # ---- Step 1: Target encoder 编码完整图像 ----
        with torch.no_grad():
            target_outputs = self.target_encoder(pixel_values)
            if hasattr(target_outputs, "last_hidden_state"):
                target_features = target_outputs.last_hidden_state
            elif isinstance(target_outputs, torch.Tensor):
                target_features = target_outputs
            else:
                target_features = target_outputs[0]
            target_features = target_features.detach()  # (B, N, D)

        # ---- Step 2: 生成文本条件化的空间掩码 ----
        mask = self.mask_generator.generate(texts, bboxes, grid_h, grid_w)
        mask = mask.to(device)  # (B, N) bool

        # ---- Step 3: Context encoder 编码被遮挡的图像 + 文本 KV ----
        # 获取 patch embeddings
        vit = self.context_encoder.vit_encoder
        if hasattr(vit, "embeddings"):
            patch_embeddings = vit.embeddings(pixel_values)  # SigLIP
        elif hasattr(vit, "patch_embed"):
            patch_embeddings = vit.patch_embed(pixel_values)  # timm
        else:
            raise ValueError("Cannot find patch embedding in ViT")

        # 遮挡位置替换为 mask_embed
        mask_expanded = mask.unsqueeze(-1).expand_as(patch_embeddings)
        masked_embeddings = patch_embeddings.clone()
        masked_embeddings[mask_expanded] = self.mask_embed.expand(B, patch_embeddings.shape[1], -1)[mask_expanded]

        # 通过 context encoder（monkey-patched attention 会注入文本 KV）
        context_features = self.context_encoder.forward_from_embeddings(
            patch_embeddings=masked_embeddings,
            text_tokens=text_tokens,
            text_padding_mask=text_padding_mask,
        )  # (B, N, D)

        # ---- Step 4: 获取文本 context（给 predictor 用）----
        text_context = self.context_encoder.text_context_encoder(text_tokens)
        if text_padding_mask is not None:
            text_context = text_context.masked_fill(text_padding_mask.unsqueeze(-1), 0.0)

        # ---- Step 5: Predictor 预测遮挡位置的特征 ----
        predictions = self.predictor(
            context_features=context_features,
            mask=mask,
            text_context=text_context,
            text_padding_mask=text_padding_mask,
        )  # (B, N_masked, D)

        # 提取 target 在遮挡位置的特征
        targets = target_features[mask].view(B, -1, target_features.shape[-1])

        # ---- Step 6: 计算 loss ----
        loss = self.compute_loss(predictions, targets)

        return {
            "loss": loss,
            "predictions": predictions,
            "targets": targets,
            "mask": mask,
        }


# ============================================================
# 4. 预训练权重迁移工具
# ============================================================

def load_pretrained_kv_injection(
    spatial_vlm: nn.Module,
    pretrain_checkpoint: str,
    strict: bool = False,
) -> None:
    """将预训练的 KV Injection 权重加载到 SpatialVLM。

    只迁移 TextContextEncoder + KVInjectionHeads，
    丢弃 LatentPredictor 和 target_encoder。

    Args:
        spatial_vlm: 目标 SpatialVLM 模型
        pretrain_checkpoint: 预训练 checkpoint 路径
        strict: 是否要求严格匹配
    """
    state_dict = torch.load(pretrain_checkpoint, map_location="cpu")

    # 提取 context encoder 的权重
    prefix = "context_encoder."
    kv_state_dict = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix) and not k.startswith(prefix + "vit_encoder.")
    }

    # 加载到 SpatialVLM 的 visual_encoder（TextConditionedViTKV）
    visual_encoder = spatial_vlm.visual_encoder
    missing, unexpected = visual_encoder.load_state_dict(kv_state_dict, strict=strict)

    if missing:
        print(f"[load_pretrained_kv_injection] Missing keys: {missing}")
    if unexpected:
        print(f"[load_pretrained_kv_injection] Unexpected keys: {unexpected}")
    print(f"[load_pretrained_kv_injection] Loaded {len(kv_state_dict)} parameter tensors")
