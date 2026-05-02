from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import AVVQCRDataset
from src.model import QCRQwenModel

from .common import ensure_output_dir, load_config, set_global_seed


def run_sft_stage(config_path: str | Path) -> None:
    config = load_config(config_path)
    set_global_seed(config["project"]["seed"])
    output_dir = ensure_output_dir(config, config["stage"])
    model = QCRQwenModel.from_config(config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    train_dataset = AVVQCRDataset(
        manifest_paths=[config["data"]["train_manifest"]],
        max_samples=config["data"].get("max_train_samples"),
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["per_device_train_batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    grad_acc_steps = config["training"]["gradient_accumulation_steps"]
    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config["training"]["num_train_epochs"]):
        for batch in dataloader:
            for sample in batch:
                losses = model.forward_sample(sample)
                loss = config["loss"]["lambda_final"] * losses["final_loss"]
                if config["loss"].get("use_draft_loss", True) and "draft_loss" in losses:
                    loss = loss + config["loss"]["lambda_draft"] * losses["draft_loss"]
                (loss / grad_acc_steps).backward()
                step += 1
                if step % grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step % config["training"]["logging_steps"] == 0:
                    draft_loss = losses.get("draft_loss")
                    draft_text = f"{draft_loss.item():.4f}" if draft_loss is not None else "n/a"
                    print(
                        f"[qcr] epoch={epoch} step={step} "
                        f"loss={loss.item():.4f} "
                        f"draft={draft_text} "
                        f"final={losses['final_loss'].item():.4f}"
                    )

    if step % grad_acc_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    save_dir = output_dir / "checkpoint-final"
    save_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.base_model, "save_pretrained"):
        model.base_model.save_pretrained(str(save_dir))
    if hasattr(model.processor, "save_pretrained"):
        model.processor.save_pretrained(str(save_dir))
