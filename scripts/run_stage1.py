"""
阶段 1：VLM 对齐预训练

冻结 ViT backbone 和 LLM，只训练新增模块：
- TextContextEncoder + KVInjectionHeads（方案 A，可从阶段 0 加载预训练权重）
- HybridPositionEmbedding（方案 B）
- LatentReasoningLoop（方案 C）
- Visual Projector（MLP）

数据：LLaVA-Instruct-150K 或类似的 image-text 对齐数据

使用方式：
    python scripts/run_stage1.py --config configs/default.yaml
    python scripts/run_stage1.py \
        --data_root data/llava_instruct \
        --annotation data/llava_instruct/llava_instruct_150k.json \
        --pretrain_weights checkpoints/pretrain/kv_injection_pretrained.pt
"""

import argparse
import os
import sys
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.spatial_vlm.spatial_vlm import SpatialVLM
from src.data.sft_dataset import SFTDataset, sft_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="阶段 1: VLM 对齐预训练")

    # 数据
    parser.add_argument("--data_root", type=str, default="data/llava_instruct", help="数据根目录")
    parser.add_argument("--annotation", type=str, default=None,
                        help="标注文件路径，默认 data_root/llava_instruct_150k.json")
    parser.add_argument("--data_format", type=str, default="llava", choices=["llava", "spatial_qa"])
    parser.add_argument("--num_workers", type=int, default=8)

    # 模型
    parser.add_argument("--vit_name", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--pretrain_weights", type=str, default=None,
                        help="阶段 0 预训练的 KV Injection 权重路径")

    # 方案开关
    parser.add_argument("--enable_kv_injection", action="store_true", default=True)
    parser.add_argument("--enable_spatial_rope", action="store_true", default=True)
    parser.add_argument("--enable_latent_reasoning", action="store_true", default=True)

    # 训练
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048, help="文本最大 token 长度")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # 输出
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage1")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1)

    # 配置文件
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def freeze_module(module: nn.Module) -> None:
    """冻结模块的所有参数。"""
    for param in module.parameters():
        param.requires_grad = False


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()

    # 配置文件
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
        stage1_config = config.get("training", {}).get("stage1", {})
        if args.epochs == 1 and "epochs" in stage1_config:
            args.epochs = stage1_config["epochs"]
        if args.batch_size == 32 and "batch_size" in stage1_config:
            args.batch_size = stage1_config["batch_size"]
        if args.learning_rate == 1e-3 and "learning_rate" in stage1_config:
            args.learning_rate = stage1_config["learning_rate"]

    if args.annotation is None:
        args.annotation = os.path.join(args.data_root, "llava_instruct_150k.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 加载模型组件 ----
    print("\n加载模型组件...")

    # ViT
    from transformers import SiglipVisionModel
    vit_encoder = SiglipVisionModel.from_pretrained(args.vit_name)
    visual_dim = vit_encoder.config.hidden_size
    vit_num_heads = vit_encoder.config.num_attention_heads

    # LLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"加载 LLM: {args.llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_decoder = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    llm_dim = llm_decoder.config.hidden_size
    llm_num_heads = llm_decoder.config.num_attention_heads

    # Text encoder: 使用 LLM 本身作为 text encoder
    text_encoder = llm_decoder.model  # LLaMA 的 transformer body

    # ---- 构建 SpatialVLM ----
    print("\n构建 SpatialVLM...")
    image_size = vit_encoder.config.image_size
    patch_size = vit_encoder.config.patch_size
    grid_size = image_size // patch_size

    model = SpatialVLM(
        vit_encoder=vit_encoder,
        llm_decoder=llm_decoder,
        text_encoder=text_encoder,
        visual_dim=visual_dim,
        text_dim=llm_dim,
        llm_dim=llm_dim,
        num_heads=llm_num_heads,
        grid_height=grid_size,
        grid_width=grid_size,
        enable_text_conditioned_vit=args.enable_kv_injection,
        enable_spatial_rope=args.enable_spatial_rope,
        enable_spatial_cross_attention=False,
        enable_latent_reasoning_loop=args.enable_latent_reasoning,
        scheme_a_mode="kv_injection",
        context_dim=256,
    )

    # 加载阶段 0 预训练权重
    if args.pretrain_weights and os.path.exists(args.pretrain_weights):
        print(f"\n加载预训练 KV Injection 权重: {args.pretrain_weights}")
        from src.training.pretrain_spatial import load_pretrained_kv_injection
        load_pretrained_kv_injection(model, args.pretrain_weights, strict=False)

    # ---- 冻结策略 ----
    print("\n设置冻结策略...")

    # 冻结 ViT backbone（KV Injection heads 保持可训练）
    if hasattr(model.visual_encoder, "vit_encoder"):
        freeze_module(model.visual_encoder.vit_encoder)
        print("  ViT backbone: 冻结")
    else:
        freeze_module(model.visual_encoder)
        print("  ViT: 冻结")

    # 冻结 LLM
    freeze_module(model.llm_decoder)
    print("  LLM decoder: 冻结")

    # 冻结 text encoder
    freeze_module(model.text_encoder)
    print("  Text encoder: 冻结")

    # 可训练模块列表
    trainable_modules = []
    if args.enable_kv_injection and hasattr(model.visual_encoder, "text_context_encoder"):
        trainable_modules.append(("TextContextEncoder", model.visual_encoder.text_context_encoder))
        trainable_modules.append(("KVInjectionHeads", model.visual_encoder.kv_injection_heads))
    if args.enable_latent_reasoning and hasattr(model, "latent_reasoning_loop"):
        trainable_modules.append(("LatentReasoningLoop", model.latent_reasoning_loop))
    if args.enable_spatial_rope and hasattr(model, "hybrid_position_embedding"):
        trainable_modules.append(("HybridPositionEmbedding", model.hybrid_position_embedding))
    trainable_modules.append(("VisualProjector", model.visual_projector))
    if hasattr(model, "text_projector") and not isinstance(model.text_projector, nn.Identity):
        trainable_modules.append(("TextProjector", model.text_projector))

    for name, module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  {name}: 可训练 ({param_count / 1e6:.2f}M)")

    params = count_parameters(model)
    print(f"\n参数统计:")
    print(f"  总参数: {params['total'] / 1e6:.1f}M")
    print(f"  可训练: {params['trainable'] / 1e6:.1f}M")
    print(f"  冻结:   {params['frozen'] / 1e6:.1f}M")

    model = model.to(device)

    # Patch LLM RoPE（方案 B）
    if args.enable_spatial_rope:
        model.patch_llm_with_spatial_rope()
        print("  已 patch LLM RoPE → HybridPositionEmbedding")

    # ---- 数据集 ----
    print(f"\n构建数据集: {args.annotation}")
    train_dataset = SFTDataset(
        data_root=args.data_root,
        annotation_file=args.annotation,
        tokenizer=tokenizer,
        image_size=image_size,
        max_length=args.max_length,
        data_format=args.data_format,
    )
    print(f"训练样本数: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=sft_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ---- 优化器 ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler = GradScaler(enabled=args.use_amp)

    # ---- 恢复训练 ----
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n从 checkpoint 恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 训练循环 ----
    print(f"\n{'='*60}")
    print(f"  阶段 1：VLM 对齐预训练")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} × {args.gradient_accumulation_steps} (accum)")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")

    model.train()
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start_time = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast(enabled=args.use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            actual_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss += actual_loss
            epoch_steps += 1

            if global_step > 0 and global_step % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start_time
                steps_per_sec = epoch_steps / elapsed

                print(
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Step {global_step} | "
                    f"Loss: {actual_loss:.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

        # Epoch 结束
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        epoch_time = time.time() - epoch_start_time
        print(
            f"\n  Epoch {epoch+1} 完成 | Loss: {avg_epoch_loss:.4f} | 耗时: {epoch_time/60:.1f} min\n"
        )

        # 保存
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"stage1_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint: {checkpoint_path}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = os.path.join(args.output_dir, "best_stage1.pt")
                torch.save({"model_state_dict": model.state_dict(), "loss": best_loss}, best_path)
                print(f"Best model: {best_path}")

    print("\n阶段 1 完成！")
    print(f"下一步：运行阶段 2 全量微调")
    print(f"  python scripts/run_stage2.py --stage1_weights {os.path.join(args.output_dir, 'best_stage1.pt')}")


if __name__ == "__main__":
    main()
