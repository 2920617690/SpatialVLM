"""
阶段 2：全量微调

解冻所有模块（ViT + LLM + 新增模块），使用空间推理数据进行端到端微调。
使用较小的学习率，防止灾难性遗忘。

数据混合：
- 空间推理数据（InternSpatial / SpaRE / SpatialRGPT）
- 通用 VQA 数据（LLaVA-Instruct，防止遗忘）

使用方式：
    python scripts/run_stage2.py --config configs/default.yaml
    python scripts/run_stage2.py \
        --stage1_weights checkpoints/stage1/best_stage1.pt \
        --data_root data/spatial_qa \
        --annotation data/spatial_qa/train.json
"""

import argparse
import os
import sys
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.spatial_vlm.spatial_vlm import SpatialVLM
from src.data.sft_dataset import SFTDataset, sft_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="阶段 2: 全量微调")

    # 数据
    parser.add_argument("--data_root", type=str, default="/primus_datasets/external_data/edu/mllm/fwk/vlm/spatial_qa", help="空间推理数据根目录")
    parser.add_argument("--annotation", type=str, default=None, help="空间推理标注文件")
    parser.add_argument("--general_data_root", type=str, default=None, help="通用 VQA 数据根目录（防遗忘）")
    parser.add_argument("--general_annotation", type=str, default=None, help="通用 VQA 标注文件")
    parser.add_argument("--data_format", type=str, default="spatial_qa", choices=["llava", "spatial_qa"])
    parser.add_argument("--num_workers", type=int, default=8)

    # 模型
    parser.add_argument("--vit_name", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--stage1_weights", type=str, required=True, help="阶段 1 的 checkpoint 路径")

    # 方案开关
    parser.add_argument("--enable_kv_injection", action="store_true", default=True)
    parser.add_argument("--enable_spatial_rope", action="store_true", default=True)
    parser.add_argument("--enable_latent_reasoning", action="store_true", default=True)

    # 训练
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--vit_lr_scale", type=float, default=0.1, help="ViT 学习率缩放因子")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # 输出
    parser.add_argument("--output_dir", type=str, default="/primus_datasets/external_data/edu/mllm/fwk/vlm/checkpoints/stage2")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1)

    # 配置文件
    parser.add_argument("--config", type=str, default=None)

    return parser.parse_args()


def build_param_groups(model: SpatialVLM, base_lr: float, vit_lr_scale: float) -> list:
    """构建分层学习率的参数组。

    - ViT backbone: base_lr * vit_lr_scale（较小学习率保护预训练特征）
    - LLM: base_lr（标准学习率）
    - 新增模块: base_lr（标准学习率）
    """
    vit_params = []
    llm_params = []
    new_module_params = []

    # 分类参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "vit_encoder" in name and "text_context_encoder" not in name and "kv_injection" not in name:
            vit_params.append(param)
        elif "llm_decoder" in name:
            llm_params.append(param)
        else:
            new_module_params.append(param)

    param_groups = []
    if vit_params:
        param_groups.append({"params": vit_params, "lr": base_lr * vit_lr_scale, "name": "vit"})
    if llm_params:
        param_groups.append({"params": llm_params, "lr": base_lr, "name": "llm"})
    if new_module_params:
        param_groups.append({"params": new_module_params, "lr": base_lr, "name": "new_modules"})

    return param_groups


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
        stage2_config = config.get("training", {}).get("stage2", {})
        if args.epochs == 1 and "epochs" in stage2_config:
            args.epochs = stage2_config["epochs"]
        if args.learning_rate == 2e-5 and "learning_rate" in stage2_config:
            args.learning_rate = stage2_config["learning_rate"]

    if args.annotation is None:
        args.annotation = os.path.join(args.data_root, "train.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 加载模型 ----
    print("\n加载模型组件...")

    from transformers import SiglipVisionModel, AutoModelForCausalLM, AutoTokenizer

    vit_encoder = SiglipVisionModel.from_pretrained(args.vit_name)
    visual_dim = vit_encoder.config.hidden_size
    vit_num_heads = vit_encoder.config.num_attention_heads

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
    text_encoder = llm_decoder.model

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

    # 加载阶段 1 权重
    print(f"\n加载阶段 1 权重: {args.stage1_weights}")
    checkpoint = torch.load(args.stage1_weights, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # ---- 阶段 2：解冻所有参数 ----
    print("\n解冻所有参数...")
    for param in model.parameters():
        param.requires_grad = True

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {params / 1e9:.2f}B")

    model = model.to(device)

    if args.enable_spatial_rope:
        model.patch_llm_with_spatial_rope()
        print("已 patch LLM RoPE")

    # ---- 数据集 ----
    print(f"\n构建数据集...")
    datasets = []

    # 空间推理数据
    spatial_dataset = SFTDataset(
        data_root=args.data_root,
        annotation_file=args.annotation,
        tokenizer=tokenizer,
        image_size=image_size,
        max_length=args.max_length,
        data_format=args.data_format,
    )
    datasets.append(spatial_dataset)
    print(f"  空间推理数据: {len(spatial_dataset)} 条")

    # 通用 VQA 数据（防遗忘）
    if args.general_data_root and args.general_annotation:
        if os.path.exists(args.general_annotation):
            general_dataset = SFTDataset(
                data_root=args.general_data_root,
                annotation_file=args.general_annotation,
                tokenizer=tokenizer,
                image_size=image_size,
                max_length=args.max_length,
                data_format="llava",
            )
            datasets.append(general_dataset)
            print(f"  通用 VQA 数据: {len(general_dataset)} 条")

    if len(datasets) > 1:
        train_dataset = ConcatDataset(datasets)
    else:
        train_dataset = datasets[0]
    print(f"  总训练样本: {len(train_dataset)} 条")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=sft_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ---- 优化器（分层学习率）----
    param_groups = build_param_groups(model, args.learning_rate, args.vit_lr_scale)
    for group in param_groups:
        param_count = sum(p.numel() for p in group["params"])
        print(f"  参数组 [{group['name']}]: {param_count/1e6:.1f}M params, lr={group['lr']:.2e}")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler = GradScaler(enabled=args.use_amp)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 训练循环 ----
    print(f"\n{'='*60}")
    print(f"  阶段 2：全量微调")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Base LR: {args.learning_rate}, ViT LR: {args.learning_rate * args.vit_lr_scale}")
    print(f"{'='*60}\n")

    model.train()
    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

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

        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        epoch_time = time.time() - epoch_start_time
        print(f"\n  Epoch {epoch+1} 完成 | Loss: {avg_epoch_loss:.4f} | 耗时: {epoch_time/60:.1f} min\n")

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"stage2_epoch{epoch+1}.pt")
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
                best_path = os.path.join(args.output_dir, "best_stage2.pt")
                torch.save({"model_state_dict": model.state_dict(), "loss": best_loss}, best_path)
                print(f"Best model: {best_path}")

    # 恢复 LLM RoPE
    if args.enable_spatial_rope:
        model.restore_llm_rope()

    print("\n阶段 2 完成！")
    print(f"最终模型: {os.path.join(args.output_dir, 'best_stage2.pt')}")
    print(f"下一步：运行评测")
    print(f"  python scripts/evaluate.py --model_path {os.path.join(args.output_dir, 'best_stage2.pt')}")


if __name__ == "__main__":
    main()
