#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.builders.coco_relations import build_coco_relation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build relation-level AVV training data.")
    parser.add_argument("--instances", type=Path, required=True, help="COCO instances json path.")
    parser.add_argument("--output", type=Path, required=True, help="Output jsonl manifest path.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional image root override.")
    parser.add_argument("--min-area", type=float, default=32.0, help="Minimum box area.")
    parser.add_argument("--center-margin", type=float, default=0.05, help="Minimum normalized center separation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = build_coco_relation_manifest(
        instances_path=args.instances,
        output_path=args.output,
        image_root=args.image_root,
        min_area=args.min_area,
        center_margin=args.center_margin,
    )
    print(f"Wrote {samples} samples to {args.output}")


if __name__ == "__main__":
    main()
