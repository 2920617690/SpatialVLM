#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize AVV training data.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.data.synthetic_generator import generate_synthetic_dataset
    from src.train.common import load_config

    config = load_config(args.config)
    generate_synthetic_dataset(config)


if __name__ == "__main__":
    main()
