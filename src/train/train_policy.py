from __future__ import annotations

from pathlib import Path

from src.model import QwenAVVAgent
from src.rl import BudgetedVerificationPolicyOptimizer

from .common import ensure_output_dir, load_config, set_global_seed


def run_policy_stage(config_path: str | Path) -> None:
    config = load_config(config_path)
    set_global_seed(config["project"]["seed"])
    output_dir = ensure_output_dir(config, config["stage"])
    agent = QwenAVVAgent.from_config(config)
    optimizer = BudgetedVerificationPolicyOptimizer(
        agent=agent,
        config=config,
        output_dir=output_dir,
    )
    optimizer.train()
