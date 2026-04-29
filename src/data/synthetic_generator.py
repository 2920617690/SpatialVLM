from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from .schema import AVVSample, SceneObject, SubClaim, TrajectoryStep, save_avv_samples


COLOR_MAP = {
    "red": (226, 74, 51),
    "blue": (66, 135, 245),
    "green": (67, 160, 71),
    "yellow": (244, 197, 66),
    "purple": (142, 68, 173),
    "orange": (239, 125, 32),
    "pink": (232, 100, 161),
    "gray": (120, 120, 120),
}

SHAPES = ["circle", "square", "triangle", "diamond"]
SIZES = ["small", "medium", "large"]


def _normalized_box(x1: int, y1: int, x2: int, y2: int, image_size: int) -> List[float]:
    return [x1 / image_size, y1 / image_size, x2 / image_size, y2 / image_size]


def _center_from_box(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _box_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _relation_left_of(a: SceneObject, b: SceneObject, margin: float = 0.04) -> bool:
    return a.center[0] + margin < b.center[0]


def _relation_right_of(a: SceneObject, b: SceneObject, margin: float = 0.04) -> bool:
    return a.center[0] > b.center[0] + margin


def _relation_above(a: SceneObject, b: SceneObject, margin: float = 0.04) -> bool:
    return a.center[1] + margin < b.center[1]


def _relation_below(a: SceneObject, b: SceneObject, margin: float = 0.04) -> bool:
    return a.center[1] > b.center[1] + margin


def _distance(a: SceneObject, b: SceneObject) -> float:
    return math.dist(a.center, b.center)


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
    if shape == "circle":
        draw.ellipse(bbox, fill=color, outline=(20, 20, 20), width=3)
        return
    if shape == "square":
        draw.rectangle(bbox, fill=color, outline=(20, 20, 20), width=3)
        return
    x1, y1, x2, y2 = bbox
    if shape == "triangle":
        draw.polygon([(x1 + x2) // 2, y1, x1, y2, x2, y2], fill=color, outline=(20, 20, 20))
        return
    if shape == "diamond":
        draw.polygon([(x1 + x2) // 2, y1, x1, (y1 + y2) // 2, (x1 + x2) // 2, y2, x2, (y1 + y2) // 2], fill=color, outline=(20, 20, 20))
        return
    raise ValueError(f"Unknown shape: {shape}")


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _scene_summary(objects: Sequence[SceneObject]) -> str:
    return ", ".join(obj.descriptor for obj in objects)


def _make_subclaim(index: int, text: str, label: str, predicate: str, arguments: List[str]) -> SubClaim:
    return SubClaim(
        claim_id=f"claim_{index}",
        text=text,
        label=label,
        predicate=predicate,
        arguments=arguments,
    )


def _build_responses(question: str, answer: str, claim_text: str, subclaims: List[SubClaim]) -> Tuple[str, str, str]:
    draft = _json_dumps(
        {
            "action": "PROPOSE",
            "draft_answer": answer,
            "draft_claim": claim_text,
            "subclaims": [item.text for item in subclaims],
        }
    )
    overall_verdict = "support" if all(item.label == "support" for item in subclaims) else "contradict"
    verify = _json_dumps(
        {
            "action": "VERIFY",
            "overall_verdict": overall_verdict,
            "checked_claims": [{"claim_id": item.claim_id, "text": item.text, "verdict": item.label} for item in subclaims],
        }
    )
    final = _json_dumps(
        {
            "action": "ANSWER",
            "final_answer": answer,
            "short_explanation": f"Question: {question} | overall_verdict: {overall_verdict}",
        }
    )
    return draft, verify, final


def _build_trajectory(overall_verdict: str) -> List[TrajectoryStep]:
    action_sequence = ["PROPOSE", "VERIFY", "ANSWER"]
    if overall_verdict == "contradict":
        action_sequence = ["PROPOSE", "VERIFY", "ANSWER"]
    return [
        TrajectoryStep(step_id=index, action=action, target_output=_json_dumps({"action": action}), comment=f"oracle action {action.lower()}")
        for index, action in enumerate(action_sequence)
    ]


def _sample_scene_objects(config: Dict[str, Any], rng: random.Random) -> List[SceneObject]:
    image_size = config["render"]["image_size"]
    min_objects = config["scene"]["min_objects"]
    max_objects = config["scene"]["max_objects"]
    min_size = config["scene"]["min_object_size"]
    max_size = config["scene"]["max_object_size"]
    overlap_limit = config["scene"]["max_overlap_ratio"]

    num_objects = rng.randint(min_objects, max_objects)
    descriptors = [(color, shape) for color in COLOR_MAP.keys() for shape in SHAPES]
    rng.shuffle(descriptors)

    placed: List[SceneObject] = []
    for index in range(num_objects):
        color, shape = descriptors[index]
        for _ in range(200):
            size = rng.randint(min_size, max_size)
            x1 = rng.randint(24, image_size - size - 24)
            y1 = rng.randint(24, image_size - size - 24)
            x2 = x1 + size
            y2 = y1 + size
            bbox = _normalized_box(x1, y1, x2, y2, image_size)
            if any(_intersection(bbox, other.bbox) > overlap_limit * min(_box_area(bbox), _box_area(other.bbox)) for other in placed):
                continue
            size_name = "small" if size < (min_size + max_size) / 2 - 8 else "medium" if size < (min_size + max_size) / 2 + 8 else "large"
            placed.append(
                SceneObject(
                    object_id=f"obj_{index}",
                    descriptor=f"{color} {shape}",
                    color=color,
                    shape=shape,
                    size=size_name,
                    bbox=bbox,
                    center=list(_center_from_box(bbox)),
                )
            )
            break
    return placed


def _render_scene(objects: Sequence[SceneObject], image_path: Path, image_size: int, background: Sequence[int]) -> None:
    image = Image.new("RGB", (image_size, image_size), tuple(background))
    draw = ImageDraw.Draw(image)
    for obj in objects:
        x1 = int(obj.bbox[0] * image_size)
        y1 = int(obj.bbox[1] * image_size)
        x2 = int(obj.bbox[2] * image_size)
        y2 = int(obj.bbox[3] * image_size)
        _draw_shape(draw, obj.shape, (x1, y1, x2, y2), COLOR_MAP[obj.color])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)


def _pick_distinct(objects: Sequence[SceneObject], rng: random.Random, k: int) -> Optional[List[SceneObject]]:
    if len(objects) < k:
        return None
    return rng.sample(list(objects), k=k)


def _make_atomic_task(objects: Sequence[SceneObject], rng: random.Random) -> Optional[Dict[str, Any]]:
    pair = _pick_distinct(objects, rng, 2)
    if pair is None:
        return None
    a, b = pair
    relations = [
        ("left_of", _relation_left_of(a, b), f"{a.descriptor} is left of {b.descriptor}", f"Is the {a.descriptor} to the left of the {b.descriptor}?"),
        ("right_of", _relation_right_of(a, b), f"{a.descriptor} is right of {b.descriptor}", f"Is the {a.descriptor} to the right of the {b.descriptor}?"),
        ("above", _relation_above(a, b), f"{a.descriptor} is above {b.descriptor}", f"Is the {a.descriptor} above the {b.descriptor}?"),
        ("below", _relation_below(a, b), f"{a.descriptor} is below {b.descriptor}", f"Is the {a.descriptor} below the {b.descriptor}?"),
    ]
    relation, truth, claim_text, question = rng.choice(relations)
    answer = "yes" if truth else "no"
    subclaims = [_make_subclaim(0, claim_text, "support" if truth else "contradict", relation, [a.object_id, b.object_id])]
    return {"question": question, "answer": answer, "claim_text": claim_text, "subclaims": subclaims}


def _make_conjunction_task(objects: Sequence[SceneObject], rng: random.Random) -> Optional[Dict[str, Any]]:
    triple = _pick_distinct(objects, rng, 3)
    if triple is None:
        return None
    a, b, c = triple
    claim1 = _relation_left_of(a, b)
    claim2 = _relation_above(a, c)
    question = f"Is the {a.descriptor} to the left of the {b.descriptor} and above the {c.descriptor}?"
    subclaims = [
        _make_subclaim(0, f"{a.descriptor} is left of {b.descriptor}", "support" if claim1 else "contradict", "left_of", [a.object_id, b.object_id]),
        _make_subclaim(1, f"{a.descriptor} is above {c.descriptor}", "support" if claim2 else "contradict", "above", [a.object_id, c.object_id]),
    ]
    answer = "yes" if claim1 and claim2 else "no"
    claim_text = f"{a.descriptor} is left of {b.descriptor} and above {c.descriptor}"
    return {"question": question, "answer": answer, "claim_text": claim_text, "subclaims": subclaims}


def _make_reference_task(objects: Sequence[SceneObject], rng: random.Random) -> Optional[Dict[str, Any]]:
    pair = _pick_distinct(objects, rng, 2)
    if pair is None:
        return None
    target, anchor = pair
    candidates = [obj for obj in objects if obj.object_id != anchor.object_id and _relation_left_of(obj, anchor)]
    if len(candidates) != 1:
        return None
    referenced = candidates[0]
    truth = referenced.object_id == target.object_id
    proposed_descriptor = referenced.descriptor if rng.random() < 0.5 else target.descriptor
    truth = referenced.descriptor == proposed_descriptor
    question = f"Is the object to the left of the {anchor.descriptor} the {proposed_descriptor}?"
    subclaims = [
        _make_subclaim(0, f"There is a unique object left of {anchor.descriptor}", "support", "exists_unique_left_of", [anchor.object_id]),
        _make_subclaim(1, f"The unique object left of {anchor.descriptor} is {proposed_descriptor}", "support" if truth else "contradict", "descriptor_match", [referenced.object_id]),
    ]
    answer = "yes" if truth else "no"
    claim_text = f"The object left of {anchor.descriptor} is {proposed_descriptor}"
    return {"question": question, "answer": answer, "claim_text": claim_text, "subclaims": subclaims}


def _make_compare_distance_task(objects: Sequence[SceneObject], rng: random.Random) -> Optional[Dict[str, Any]]:
    triple = _pick_distinct(objects, rng, 3)
    if triple is None:
        return None
    a, b, ref = triple
    truth = _distance(a, ref) < _distance(b, ref)
    question = f"Is the {a.descriptor} closer to the {ref.descriptor} than the {b.descriptor} is?"
    claim_text = f"{a.descriptor} is closer to {ref.descriptor} than {b.descriptor}"
    subclaims = [_make_subclaim(0, claim_text, "support" if truth else "contradict", "closer_than", [a.object_id, b.object_id, ref.object_id])]
    answer = "yes" if truth else "no"
    return {"question": question, "answer": answer, "claim_text": claim_text, "subclaims": subclaims}


def _make_count_task(objects: Sequence[SceneObject], rng: random.Random) -> Optional[Dict[str, Any]]:
    pair = _pick_distinct(objects, rng, 2)
    if pair is None:
        return None
    _, anchor = pair
    matches = [obj for obj in objects if obj.object_id != anchor.object_id and _relation_left_of(obj, anchor)]
    answer = str(len(matches))
    question = f"How many objects are to the left of the {anchor.descriptor}?"
    claim_text = f"The number of objects left of {anchor.descriptor} is {answer}"
    subclaims = [
        _make_subclaim(index, f"{obj.descriptor} is left of {anchor.descriptor}", "support", "left_of", [obj.object_id, anchor.object_id])
        for index, obj in enumerate(matches)
    ]
    if not subclaims:
        subclaims = [_make_subclaim(0, f"No object is left of {anchor.descriptor}", "support", "count_zero", [anchor.object_id])]
    return {"question": question, "answer": answer, "claim_text": claim_text, "subclaims": subclaims}


def _make_chain_task(objects: Sequence[SceneObject], rng: random.Random) -> Optional[Dict[str, Any]]:
    triple = _pick_distinct(objects, rng, 3)
    if triple is None:
        return None
    _, anchor, ref = triple
    left_candidates = [obj for obj in objects if obj.object_id not in {anchor.object_id, ref.object_id} and _relation_left_of(obj, anchor)]
    if len(left_candidates) != 1:
        return None
    target = left_candidates[0]
    truth = _relation_above(target, ref)
    question = f"Is the object to the left of the {anchor.descriptor} above the {ref.descriptor}?"
    subclaims = [
        _make_subclaim(0, f"The unique object left of {anchor.descriptor} is {target.descriptor}", "support", "resolve_left_of", [target.object_id, anchor.object_id]),
        _make_subclaim(1, f"{target.descriptor} is above {ref.descriptor}", "support" if truth else "contradict", "above", [target.object_id, ref.object_id]),
    ]
    answer = "yes" if truth else "no"
    claim_text = f"The object left of {anchor.descriptor} is above {ref.descriptor}"
    return {"question": question, "answer": answer, "claim_text": claim_text, "subclaims": subclaims}


TASK_BUILDERS = {
    "atomic_yesno": _make_atomic_task,
    "conjunction_yesno": _make_conjunction_task,
    "reference_yesno": _make_reference_task,
    "compare_distance_yesno": _make_compare_distance_task,
    "count_relation": _make_count_task,
    "chain_relation_yesno": _make_chain_task,
}


def _choose_task_name(config: Dict[str, Any], rng: random.Random) -> str:
    task_items = list(config["tasks"].items())
    total = sum(weight for _, weight in task_items)
    threshold = rng.random() * total
    running = 0.0
    for name, weight in task_items:
        running += weight
        if running >= threshold:
            return name
    return task_items[-1][0]


def _generate_single_sample(
    split: str,
    index: int,
    config: Dict[str, Any],
    rng: random.Random,
) -> AVVSample:
    objects = _sample_scene_objects(config, rng)
    task_name = _choose_task_name(config, rng)

    task_payload = None
    for _ in range(32):
        builder = TASK_BUILDERS[task_name]
        task_payload = builder(objects, rng)
        if task_payload is not None:
            break
        task_name = _choose_task_name(config, rng)
    if task_payload is None:
        raise RuntimeError("Failed to generate a valid task payload.")

    draft_response, verify_response, final_response = _build_responses(
        question=task_payload["question"],
        answer=task_payload["answer"],
        claim_text=task_payload["claim_text"],
        subclaims=task_payload["subclaims"],
    )
    verify_json = json.loads(verify_response)
    trajectory = _build_trajectory(verify_json["overall_verdict"])
    image_rel_path = Path("images") / split / f"{split}_{index:06d}.png"
    return AVVSample(
        sample_id=f"{split}_{index:06d}",
        split=split,
        image_path=str(image_rel_path),
        task_type=task_name,
        question=task_payload["question"],
        answer=str(task_payload["answer"]),
        draft_response=draft_response,
        verify_response=verify_response,
        final_response=final_response,
        subclaims=task_payload["subclaims"],
        trajectory=trajectory,
        scene_objects=list(objects),
        metadata={
            "scene_summary": _scene_summary(objects),
            "overall_verdict": verify_json["overall_verdict"],
        },
    )


def _write_summary(samples: Sequence[AVVSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_counter = Counter(sample.task_type for sample in samples)
    answer_counter = Counter(sample.answer for sample in samples)
    payload = {
        "num_samples": len(samples),
        "task_counts": dict(task_counter),
        "answer_counts": dict(answer_counter),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_synthetic_dataset(config: Dict[str, Any]) -> None:
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config["seed"])
    image_size = config["render"]["image_size"]
    background = config["render"]["background"]

    for split, count in config["splits"].items():
        samples: List[AVVSample] = []
        image_dir = output_root / "images" / split
        for index in range(count):
            sample = _generate_single_sample(split, index, config, rng)
            image_path = output_root / sample.image_path
            _render_scene(sample.scene_objects, image_path, image_size=image_size, background=background)
            sample.image_path = str(image_path)
            samples.append(sample)

        manifest_path = output_root / "manifests" / f"{split}.jsonl"
        save_avv_samples(samples, manifest_path)
        _write_summary(samples, output_root / "metadata" / f"{split}_summary.json")
        print(f"[synthesize] wrote {len(samples)} samples to {manifest_path}")
