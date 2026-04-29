from __future__ import annotations

from pathlib import Path

from transformers import Trainer, TrainingArguments

from src.data import AVVSupervisedDataset, QwenChatCollator
from src.model import load_qwen_components

from .common import ensure_output_dir, load_config, set_global_seed


def run_sft_stage(config_path: str | Path) -> None:
    config = load_config(config_path)
    set_global_seed(config["project"]["seed"])
    output_dir = ensure_output_dir(config, config["stage"])
    components = load_qwen_components(config)

    train_dataset = AVVSupervisedDataset(
        manifest_paths=[config["data"]["train_manifest"]],
        modes=config["dataset"]["modes"],
        max_samples=config["data"].get("max_train_samples"),
    )
    eval_dataset = AVVSupervisedDataset(
        manifest_paths=[config["data"]["val_manifest"]],
        modes=config["dataset"]["modes"],
        max_samples=config["data"].get("max_eval_samples"),
    )
    collator = QwenChatCollator(
        processor=components.processor,
        max_length=config["model"]["max_prompt_length"],
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        num_train_epochs=config["training"]["num_train_epochs"],
        warmup_ratio=config["training"]["warmup_ratio"],
        logging_steps=config["training"]["logging_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],
        remove_unused_columns=config["training"]["remove_unused_columns"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        report_to=config["training"]["report_to"],
        evaluation_strategy="steps",
        save_strategy="steps",
    )

    trainer = Trainer(
        model=components.model,
        args=args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "checkpoint-final"))
    if hasattr(components.processor, "save_pretrained"):
        components.processor.save_pretrained(str(output_dir / "checkpoint-final"))
