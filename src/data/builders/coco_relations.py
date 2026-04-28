from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.data.schema import RelationSample, save_relation_manifest


RELATION_QUESTION_TEMPLATES = {
    "left_of": "Is the {a} to the left of the {b}?",
    "right_of": "Is the {a} to the right of the {b}?",
    "above": "Is the {a} above the {b}?",
    "below": "Is the {a} below the {b}?",
    "overlap": "Does the {a} overlap the {b}?",
    "inside": "Is the {a} inside the {b}?",
    "larger_than": "Is the {a} larger than the {b}?",
    "smaller_than": "Is the {a} smaller than the {b}?",
}

INVERSE_RELATION = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "larger_than": "smaller_than",
    "smaller_than": "larger_than",
}


def _normalize_box(box: List[float], width: float, height: float) -> List[float]:
    x, y, w, h = box
    return [x / width, y / height, (x + w) / width, (y + h) / height]


def _center(box: List[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _detect_relations(
    box_a: List[float],
    box_b: List[float],
    center_margin: float,
) -> List[str]:
    relations: List[str] = []
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    area_a = _area(box_a)
    area_b = _area(box_b)
    inter = _intersection(box_a, box_b)

    if ax + center_margin < bx:
        relations.append("left_of")
    if ax > bx + center_margin:
        relations.append("right_of")
    if ay + center_margin < by:
        relations.append("above")
    if ay > by + center_margin:
        relations.append("below")
    if inter > 0.0:
        relations.append("overlap")
    if area_a > 0 and inter / area_a > 0.95:
        relations.append("inside")
    if area_a > area_b * 1.2:
        relations.append("larger_than")
    if area_b > area_a * 1.2:
        relations.append("smaller_than")
    return relations


def _make_claim(object_a: str, object_b: str, relation: str, answer: str) -> str:
    relation_text = relation.replace("_", " ")
    if answer == "yes":
        return f"{object_a} {relation_text} {object_b}"
    return f"{object_a} not {relation_text} {object_b}"


def build_coco_relation_manifest(
    instances_path: str | Path,
    output_path: str | Path,
    image_root: Optional[str | Path] = None,
    min_area: float = 32.0,
    center_margin: float = 0.05,
) -> int:
    instances_path = Path(instances_path)
    with instances_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    categories = {item["id"]: item["name"] for item in payload["categories"]}
    images = {item["id"]: item for item in payload["images"]}
    image_to_annotations: Dict[int, List[dict]] = defaultdict(list)
    for ann in payload["annotations"]:
        image_to_annotations[ann["image_id"]].append(ann)

    samples: List[RelationSample] = []
    root = Path(image_root) if image_root else instances_path.parent

    for image_id, annotations in image_to_annotations.items():
        image_info = images[image_id]
        width = image_info["width"]
        height = image_info["height"]
        file_name = image_info["file_name"]
        image_path = str(root / file_name)

        normalized = []
        for ann in annotations:
            box = _normalize_box(ann["bbox"], width, height)
            if _area(box) * width * height < min_area:
                continue
            normalized.append((ann, box))

        for i, (ann_a, box_a) in enumerate(normalized):
            for ann_b, box_b in normalized[i + 1:]:
                relations = _detect_relations(box_a, box_b, center_margin=center_margin)
                if not relations:
                    continue

                object_a = categories[ann_a["category_id"]]
                object_b = categories[ann_b["category_id"]]

                for relation in relations:
                    template = RELATION_QUESTION_TEMPLATES.get(relation)
                    if template is None:
                        continue
                    positive_question = template.format(a=object_a, b=object_b)
                    samples.append(
                        RelationSample(
                            image_path=image_path,
                            question=positive_question,
                            answer="yes",
                            relation=relation,
                            object_a=object_a,
                            object_b=object_b,
                            bbox_a=box_a,
                            bbox_b=box_b,
                            verifier_label="support",
                            claim=_make_claim(object_a, object_b, relation, "yes"),
                            metadata={"image_id": image_id, "source": "coco"},
                        )
                    )

                    inverse = INVERSE_RELATION.get(relation)
                    if inverse is None:
                        continue
                    negative_question = RELATION_QUESTION_TEMPLATES[inverse].format(a=object_a, b=object_b)
                    samples.append(
                        RelationSample(
                            image_path=image_path,
                            question=negative_question,
                            answer="no",
                            relation=inverse,
                            object_a=object_a,
                            object_b=object_b,
                            bbox_a=box_a,
                            bbox_b=box_b,
                            verifier_label="contradict",
                            claim=_make_claim(object_a, object_b, inverse, "no"),
                            metadata={"image_id": image_id, "source": "coco", "counterfactual": True},
                        )
                    )

    save_relation_manifest(samples, output_path)
    return len(samples)
