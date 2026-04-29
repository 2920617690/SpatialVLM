from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .prompts import (
    build_policy_prompt,
    mode_to_system_prompt,
    mode_to_target,
    mode_to_user_prompt,
    trajectory_target,
)
from .schema import AVVSample, load_avv_samples
from src.model.chat import build_messages, render_chat_text


@dataclass
class TrainingRecord:
    sample: AVVSample
    mode: str
    prompt_text: str
    target_text: str


class AVVSupervisedDataset(Dataset):
    def __init__(
        self,
        manifest_paths: Sequence[str | Path],
        modes: Sequence[str],
        max_samples: int | None = None,
    ) -> None:
        self.records: List[TrainingRecord] = []
        for manifest in manifest_paths:
            for sample in load_avv_samples(manifest):
                for mode in modes:
                    self.records.append(
                        TrainingRecord(
                            sample=sample,
                            mode=mode,
                            prompt_text=mode_to_user_prompt(sample, mode),
                            target_text=mode_to_target(sample, mode),
                        )
                    )
        if max_samples is not None:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TrainingRecord:
        return self.records[idx]


class AVVPolicyDataset(Dataset):
    def __init__(
        self,
        manifest_paths: Sequence[str | Path],
        max_samples: int | None = None,
    ) -> None:
        self.records: List[TrainingRecord] = []
        for manifest in manifest_paths:
            for sample in load_avv_samples(manifest):
                history: List[Dict[str, str]] = []
                for step in sample.trajectory:
                    prompt_text = build_policy_prompt(sample, history)
                    target_text = trajectory_target(step)
                    self.records.append(
                        TrainingRecord(
                            sample=sample,
                            mode="policy",
                            prompt_text=prompt_text,
                            target_text=target_text,
                        )
                    )
                    history.append(
                        {
                            "action": json.loads(step.target_output)["action"],
                            "observation": step.comment or "none",
                        }
                    )
        if max_samples is not None:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TrainingRecord:
        return self.records[idx]


class QwenChatCollator:
    def __init__(self, processor: Any, max_length: int = 4096) -> None:
        self.processor = processor
        self.max_length = max_length

    @staticmethod
    def _load_image(image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _encode_single(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        target_text: str,
    ) -> Dict[str, torch.Tensor]:
        prompt_messages = build_messages(system_prompt, user_prompt, assistant_text=None)
        full_messages = build_messages(system_prompt, user_prompt, assistant_text=target_text)

        prompt_text = render_chat_text(self.processor, prompt_messages, add_generation_prompt=True)
        full_text = render_chat_text(self.processor, full_messages, add_generation_prompt=False)

        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        attention_mask = full_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(full_inputs["input_ids"])
        labels[attention_mask == 0] = -100

        item: Dict[str, torch.Tensor] = {
            "input_ids": full_inputs["input_ids"].squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }
        for key, value in full_inputs.items():
            if key in item:
                continue
            item[key] = value.squeeze(0)
        return item

    def __call__(self, batch: Sequence[TrainingRecord]) -> Dict[str, Any]:
        encoded: List[Dict[str, torch.Tensor]] = []
        modes: List[str] = []
        sample_ids: List[str] = []
        for record in batch:
            image = self._load_image(record.sample.image_path)
            encoded.append(
                self._encode_single(
                    image=image,
                    system_prompt=mode_to_system_prompt(record.mode),
                    user_prompt=record.prompt_text,
                    target_text=record.target_text,
                )
            )
            modes.append(record.mode)
            sample_ids.append(record.sample.sample_id)

        result: Dict[str, Any] = {
            "input_ids": pad_sequence([item["input_ids"] for item in encoded], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "attention_mask": pad_sequence([item["attention_mask"] for item in encoded], batch_first=True, padding_value=0),
            "labels": pad_sequence([item["labels"] for item in encoded], batch_first=True, padding_value=-100),
            "modes": modes,
            "sample_ids": sample_ids,
        }

        extra_keys = [key for key in encoded[0].keys() if key not in {"input_ids", "attention_mask", "labels"}]
        for key in extra_keys:
            tensors = [item[key] for item in encoded]
            try:
                result[key] = torch.stack(tensors, dim=0)
            except RuntimeError:
                result[key] = tensors
        return result
