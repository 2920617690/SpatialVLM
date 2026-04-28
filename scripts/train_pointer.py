#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.train_pointer import run_pointer_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AVV pointer pretraining.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_pointer_stage(parse_args().config)
