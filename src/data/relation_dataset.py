from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .schema import RelationSample, save_relation_manifest


class RelationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        crop_size: int = 160,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.crop_size = crop_size
        self.samples = self._load_manifest(self.manifest_path)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.crop_transform = transforms.Compose(
            [
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def _load_manifest(path: Path) -> List[RelationSample]:
        samples: List[RelationSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                samples.append(RelationSample(**payload))
        return samples

    @staticmethod
    def save_manifest(samples: Iterable[RelationSample], output_path: str | Path) -> None:
        save_relation_manifest(samples, output_path)

    @staticmethod
    def _clamp_box(box: List[float]) -> List[float]:
        x1, y1, x2, y2 = box
        return [
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        ]

    @staticmethod
    def _crop_image(image: Image.Image, box: List[float]) -> Image.Image:
        w, h = image.size
        x1, y1, x2, y2 = RelationDataset._clamp_box(box)
        left = int(x1 * w)
        top = int(y1 * h)
        right = max(left + 1, int(x2 * w))
        bottom = max(top + 1, int(y2 * h))
        return image.crop((left, top, right, bottom))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        crop_a = self._crop_image(image, sample.bbox_a)
        crop_b = self._crop_image(image, sample.bbox_b)

        return {
            "image": self.image_transform(image),
            "crop_a": self.crop_transform(crop_a),
            "crop_b": self.crop_transform(crop_b),
            "question": sample.question,
            "answer": sample.answer,
            "relation": sample.relation,
            "claim": sample.claim or sample.question,
            "bbox_a": torch.tensor(sample.bbox_a, dtype=torch.float32),
            "bbox_b": torch.tensor(sample.bbox_b, dtype=torch.float32),
            "verifier_label": sample.verifier_label,
            "object_a": sample.object_a,
            "object_b": sample.object_b,
            "metadata": sample.metadata or {},
        }
