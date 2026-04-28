from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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


def save_relation_manifest(samples: Iterable[RelationSample], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
