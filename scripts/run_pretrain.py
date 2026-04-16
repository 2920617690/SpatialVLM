"""
阶段 0：Text-Conditioned Latent Prediction 预训练

将 V-JEPA 的隐空间预测与文本条件化 KV Injection 结合：
- Target Encoder (EMA): 原始 SigLIP ViT → 完整图像 → target features
- Context Encoder: TextConditionedViTKV → 部分可见 + 文本 KV → context features
- Predictor: context + mask tokens + 文本 cross-attn → 预测遮挡区域的 latent features
- Loss: Smooth L1 on L2-normalized features（只在遮挡位置计算）

预训练后只保留 TextContextEncoder + KVInjectionHeads 迁移到 SpatialVLM。

使用方式：
    python scripts/run_pretrain.py --config configs/default.yaml
    python scripts/run_pretrain.py --data_root data/refcoco --epochs 5 --batch_size 64
"""

import argparse
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import yaml

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.text_conditioned_vit.text_conditioned_vit_kv import TextConditionedViTKV
from src.training.pretrain_spatial import TextConditionedPretrainer
from src.data.pretrain_dataset import SpatialPretrainDataset, spatial_pretrain_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="阶段 0: Text-Conditioned Latent Prediction 预训练")

    # 数据
    parser.add_argument("--data_root", type=str, default="data/refcoco", help="RefCOCO 数据根目录")
    parser.add_argument("--precomputed_text_features", type=str, default=None,
                        help="预计算文本特征路径，默认 data_root/text_features.pt")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")

    # 模型
    parser.add_argument("--vit_name", type=str, default="google/siglip-so400m-patch14-384",
                        help="SigLIP 模型名称")
    parser.add_argument("--context_dim", type=int, default=256, help="文本 context 压缩维度")
    parser.add_argument("--predictor_dim", type=int, default=384, help="Predictor 瓶颈维度")
    parser.add_argument("--predictor_layers", type=int, default=3, help="Predictor 层数")
    parser.add_argument("--predictor_heads", type=int, default=6, help="Predictor attention heads")

    # 训练
    parser.add_argument("--epochs", type=int, default=5, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=64, help="每 GPU batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="warmup 步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--use_amp", action="store_true", default=True, help="使用混合精度")

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.996, help="EMA 初始衰减率")
    parser.add_argument("--ema_decay_end", type=float, default=1.0, help="EMA 最终衰减率")

    # 掩码
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="目标掩码比例")
    parser.add_argument("--directional_prob", type=float, default=0.5, help="方向掩码概率")
    parser.add_argument("--block_prob", type=float, default=0.25, help="Block 掩码概率")
    parser.add_argument("--random_prob", type=float, default=0.25, help="随机掩码概率")

    # 输出
    parser.add_argument("--output_dir", type=str, default="checkpoints/pretrain", help="输出目录")
    parser.add_argument("--save_every", type=int, default=1, help="每 N 个 epoch 保存一次")
    parser.add_argument("--log_every", type=int, default=50, help="每 N 步打印一次 log")

    # 配置文件（优先级低于命令行参数）
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")

    # 恢复训练
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_vit_encoder(model_name: str):
    """加载预训练 SigLIP ViT。"""
    from transformers import SiglipVisionModel
    print(f"加载 ViT: {model_name}")
    vit = SiglipVisionModel.from_pretrained(model_name)
    return vit


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
    """Cosine learning rate schedule with warmup。"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()

    # 如果指定了配置文件，用配置文件的值作为默认值
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        pretrain_config = config.get("pretraining", {})
        training_config = pretrain_config.get("training", {})
        data_config = pretrain_config.get("data", {})

        # 只覆盖未在命令行中显式指定的参数
        if args.data_root == "data/refcoco" and "data_root" in data_config:
            args.data_root = data_config["data_root"]
        if args.epochs == 5 and "epochs" in training_config:
            args.epochs = training_config["epochs"]
        if args.batch_size == 64 and "batch_size" in training_config:
            args.batch_size = training_config["batch_size"]
        if args.learning_rate == 1e-3 and "learning_rate" in training_config:
            args.learning_rate = training_config["learning_rate"]

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # 预计算文本特征路径
    if args.precomputed_text_features is None:
        args.precomputed_text_features = os.path.join(args.data_root, "text_features.pt")

    # ---- 构建数据集 ----
    print("\n构建数据集...")
    train_dataset = SpatialPretrainDataset(
        data_root=args.data_root,
        split="train",
        data_format="refcoco",
        image_size=384,
        max_text_length=77,
        precomputed_text_features=args.precomputed_text_features,
        spatial_filter=True,
        augment=True,
    )
    print(f"训练样本数: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=spatial_pretrain_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ---- 构建模型 ----
    print("\n构建模型...")

    # 加载 SigLIP ViT
    vit_encoder = build_vit_encoder(args.vit_name)
    visual_dim = vit_encoder.config.hidden_size
    num_heads = vit_encoder.config.num_attention_heads
    print(f"ViT: visual_dim={visual_dim}, num_heads={num_heads}, "
          f"layers={vit_encoder.config.num_hidden_layers}")

    # 获取文本维度
    if os.path.exists(args.precomputed_text_features):
        text_data = torch.load(args.precomputed_text_features, map_location="cpu")
        text_dim = text_data["features"].shape[-1]
        del text_data
    else:
        text_dim = 4096  # 默认 LLaMA dim
    print(f"Text dim: {text_dim}")

    # 创建 Context Encoder（TextConditionedViTKV）
    context_encoder = TextConditionedViTKV(
        vit_encoder=vit_encoder,
        text_dim=text_dim,
        context_dim=args.context_dim,
        injection_layers=None,  # 所有层注入
        num_heads=num_heads,
    )

    # 创建预训练器
    pretrainer = TextConditionedPretrainer(
        context_encoder=context_encoder,
        vit_encoder=vit_encoder,
        text_context_dim=args.context_dim,
        visual_dim=visual_dim,
        predictor_dim=args.predictor_dim,
        predictor_layers=args.predictor_layers,
        predictor_heads=args.predictor_heads,
        ema_decay=args.ema_decay,
        ema_decay_end=args.ema_decay_end,
        loss_type="smooth_l1",
        feature_norm=True,
        mask_ratio=args.mask_ratio,
        directional_prob=args.directional_prob,
        block_prob=args.block_prob,
        random_prob=args.random_prob,
    )
    pretrainer = pretrainer.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in pretrainer.parameters())
    trainable_params = sum(p.numel() for p in pretrainer.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.1f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.1f}M")

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        [p for p in pretrainer.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler = GradScaler(enabled=args.use_amp)

    # ---- 恢复训练 ----
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n从 checkpoint 恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        pretrainer.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        print(f"恢复到 epoch {start_epoch}, step {global_step}")

    # ---- 输出目录 ----
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存训练配置
    config_save_path = os.path.join(args.output_dir, "train_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # ---- 计算 patch grid 尺寸 ----
    image_size = 384
    patch_size = vit_encoder.config.patch_size
    grid_size = image_size // patch_size  # 384 / 14 = 27（SigLIP 实际是 27x27=729 patches）
    print(f"Patch grid: {grid_size}x{grid_size} = {grid_size * grid_size} patches")

    # ---- 训练循环 ----
    print(f"\n{'='*60}")
    print(f"  开始预训练")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Mask ratio: {args.mask_ratio}")
    print(f"  掩码策略: directional={args.directional_prob}, block={args.block_prob}, random={args.random_prob}")
    print(f"{'='*60}\n")

    pretrainer.train()
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start_time = time.time()

        for step, batch in enumerate(train_loader):
            # 移到 GPU
            pixel_values = batch["pixel_values"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            texts = batch["texts"]
            bboxes = batch["bboxes"]
            text_padding_mask = batch["text_padding_mask"].to(device)

            # Forward
            with autocast(enabled=args.use_amp):
                outputs = pretrainer(
                    pixel_values=pixel_values,
                    text_tokens=text_tokens,
                    texts=texts,
                    bboxes=bboxes,
                    text_padding_mask=text_padding_mask,
                    grid_h=grid_size,
                    grid_w=grid_size,
                )
                loss = outputs["loss"]

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # 梯度裁剪
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in pretrainer.parameters() if p.requires_grad],
                    args.grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # EMA 更新 target encoder
            progress = global_step / max(1, total_steps)
            pretrainer.update_target_encoder(progress)

            # 统计
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # 日志
            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start_time
                steps_per_sec = epoch_steps / elapsed
                eta_epoch = (len(train_loader) - step - 1) / max(1, steps_per_sec)

                print(
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s | "
                    f"ETA: {eta_epoch/60:.1f} min"
                )

        # Epoch 结束
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        epoch_time = time.time() - epoch_start_time
        print(
            f"\n{'='*60}\n"
            f"  Epoch {epoch+1}/{args.epochs} 完成\n"
            f"  平均 Loss: {avg_epoch_loss:.4f}\n"
            f"  耗时: {epoch_time/60:.1f} min\n"
            f"{'='*60}\n"
        )

        # 保存 checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": pretrainer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_epoch_loss,
                "args": vars(args),
            }, checkpoint_path)
            print(f"Checkpoint 已保存: {checkpoint_path}")

            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": pretrainer.state_dict(),
                    "loss": avg_epoch_loss,
                }, best_path)
                print(f"最佳模型已保存: {best_path} (loss={best_loss:.4f})")

    # ---- 导出预训练权重（只保留 KV Injection 部分）----
    print("\n导出 KV Injection 预训练权重...")
    export_path = os.path.join(args.output_dir, "kv_injection_pretrained.pt")

    # 提取 context_encoder 中非 ViT backbone 的权重
    kv_state_dict = {}
    for name, param in pretrainer.context_encoder.state_dict().items():
        if not name.startswith("vit_encoder."):
            kv_state_dict[name] = param.cpu()

    torch.save(kv_state_dict, export_path)
    print(f"KV Injection 权重已导出: {export_path}")
    print(f"包含 {len(kv_state_dict)} 个参数 tensor")

    print("\n预训练完成！")
    print(f"下一步：使用导出的权重进行阶段 1 对齐预训练")
    print(f"  python scripts/run_stage1.py --pretrain_weights {export_path}")


if __name__ == "__main__":
    main()
