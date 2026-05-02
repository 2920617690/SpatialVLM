#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import SceneObject, save_avv_samples
from src.data.synthetic_generator import _generate_single_sample, _write_summary
from src.train.common import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize Blender-light QCR training data.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _run_blender(config_path: Path, config: dict) -> None:
    blender_binary = config["blender_binary"]
    render_script = ROOT / "scripts" / "blender_light_render.py"
    subprocess.run(
        [
            blender_binary,
            "--background",
            "--python",
            str(render_script),
            "--",
            str(config_path.resolve()),
        ],
        check=True,
    )


def _scene_objects_from_metadata(metadata: dict) -> list[SceneObject]:
    objects = []
    for item in metadata["scene_objects"]:
        objects.append(
            SceneObject(
                object_id=item["object_id"],
                descriptor=item["descriptor"],
                color=item["color"],
                shape=item["shape"],
                size=item["size"],
                bbox=item["bbox"],
                center=item["center"],
            )
        )
    return objects


def _build_samples_from_scene_metadata(config: dict) -> None:
    import random

    output_root = Path(config["output"]["root"]).resolve()
    seed = config["seed"]
    split_offsets = {"train": 0, "val": 1000, "test": 2000}
    for split, expected_scene_count in config["splits"].items():
        rng = random.Random(seed + 10_000 + split_offsets.get(split, 5000))
        q_per_scene = config["questions_per_image"][split]
        samples = []
        metadata_paths = sorted((output_root / "scene_metadata" / split).glob("*.json"))
        if len(metadata_paths) < expected_scene_count:
            raise RuntimeError(
                f"Expected at least {expected_scene_count} scene metadata files for split={split}, "
                f"but found {len(metadata_paths)}."
            )
        for metadata_path in metadata_paths[:expected_scene_count]:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            objects = _scene_objects_from_metadata(metadata)
            scene_id = metadata["scene_id"]
            scene_index = int(scene_id.split("_")[-1])
            for q_index in range(q_per_scene):
                sample = _generate_single_sample(
                    split=split,
                    scene_index=scene_index,
                    question_index=q_index,
                    config=config,
                    rng=rng,
                    objects=objects,
                )
                sample.image_path = metadata["image_path"]
                sample.metadata["render_backend"] = "blender_light"
                sample.metadata["camera"] = metadata["camera"]
                samples.append(sample)

        manifest_path = output_root / "manifests" / f"{split}.jsonl"
        save_avv_samples(samples, manifest_path)
        _write_summary(samples, output_root / "metadata" / f"{split}_summary.json")
        print(
            f"[blender-light] split={split} scenes={expected_scene_count} "
            f"questions_per_scene={q_per_scene} samples={len(samples)} "
            f"manifest={manifest_path}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _run_blender(args.config, config)
    _build_samples_from_scene_metadata(config)


if __name__ == "__main__":
    main()
