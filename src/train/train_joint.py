from __future__ import annotations

from pathlib import Path

from src.model import AVVModel

from .common import load_config


def run_joint_stage(config_path: str | Path) -> None:
    config = load_config(config_path)
    model = AVVModel(
        image_channels=config["model"]["image_channels"],
        image_size=config["model"]["image_size"],
        hidden_dim=config["model"]["hidden_dim"],
        question_dim=config["model"]["question_dim"],
        vocab_size=config["model"]["vocab_size"],
        answer_classes=config["model"]["answer_classes"],
        verifier_classes=config["model"]["verifier_classes"],
        crop_size=config["model"]["crop_size"],
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"[joint] stage={config['stage']} total_params={total}")
    print(f"[joint] train_manifest={config['data']['train_manifest']}")
