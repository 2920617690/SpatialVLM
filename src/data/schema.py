from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class RelationSample:
    image_path: str
    question: str
    answer: str
    relation: str
    object_a: str
    object_b: str
    bbox_a: List[float]
    bbox_b: List[float]
    verifier_label: str = "support"
    claim: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SceneObject:
    object_id: str
    descriptor: str
    color: str
    shape: str
    size: str
    bbox: List[float]
    center: List[float]


@dataclass
class SubClaim:
    claim_id: str
    text: str
    label: str
    predicate: str
    arguments: List[str] = field(default_factory=list)


@dataclass
class TrajectoryStep:
    step_id: int
    action: str
    target_output: str
    comment: str = ""


@dataclass
class AVVSample:
    sample_id: str
    split: str
    image_path: str
    task_type: str
    question: str
    answer: str
    draft_response: str
    verify_response: str
    final_response: str
    subclaims: List[SubClaim] = field(default_factory=list)
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    scene_objects: List[SceneObject] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "AVVSample":
        return AVVSample(
            sample_id=payload["sample_id"],
            split=payload["split"],
            image_path=payload["image_path"],
            task_type=payload["task_type"],
            question=payload["question"],
            answer=str(payload["answer"]),
            draft_response=payload["draft_response"],
            verify_response=payload["verify_response"],
            final_response=payload["final_response"],
            subclaims=[SubClaim(**item) for item in payload.get("subclaims", [])],
            trajectory=[TrajectoryStep(**item) for item in payload.get("trajectory", [])],
            scene_objects=[SceneObject(**item) for item in payload.get("scene_objects", [])],
            metadata=payload.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_relation_manifest(samples: Iterable[RelationSample], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def save_avv_samples(samples: Iterable[AVVSample], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def load_avv_samples(path: str | Path) -> List[AVVSample]:
    file_path = Path(path)
    samples: List[AVVSample] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            samples.append(AVVSample.from_dict(payload))
    return samples
