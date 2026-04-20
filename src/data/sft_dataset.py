"""
SFT Dataset — 指令微调数据集（Qwen3.5 适配版）

支持两种格式：
1. LLaVA-style: {"id": ..., "image": "xxx.jpg", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
2. Spatial QA: {"image": "xxx.jpg", "question": "...", "answer": "..."}

使用 Qwen3.5 的 AutoProcessor 处理图像和文本，
通过 apply_chat_template 生成 input_ids / pixel_values / image_grid_thw 等。

用于阶段 1（对齐预训练）和阶段 2（全量微调）。
"""

import json
import os
import random
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    """指令微调数据集（Qwen3.5 适配版）。

    使用 Qwen3.5 的 processor 处理图像和文本。
    每个样本返回 processor 输出的所有字段 + labels。
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        processor,
        max_length: int = 2048,
        data_format: str = "llava",
    ):
        """
        Args:
            data_root: 数据根目录（包含 images/ 子目录）
            annotation_file: 标注文件路径（JSON）
            processor: Qwen3.5 AutoProcessor 实例
            max_length: 文本最大 token 长度
            data_format: "llava" 或 "spatial_qa"
        """
        self.data_root = data_root
        self.processor = processor
        self.max_length = max_length
        self.data_format = data_format

        with open(annotation_file) as f:
            self.annotations = json.load(f)
        print(f"加载 {len(self.annotations)} 条标注 from {annotation_file}")

    def __len__(self) -> int:
        return len(self.annotations)

    def _format_conversation(self, item: Dict[str, Any]) -> tuple[str, str]:
        """将标注转为统一的对话格式，返回 (prompt, response)。"""
        if self.data_format == "llava":
            conversations = item.get("conversations", [])
            prompt_parts = []
            response_parts = []
            for turn in conversations:
                role = turn.get("from", "")
                value = turn.get("value", "")
                # 移除 LLaVA 格式中的 <image> 占位符（由 processor 自动处理）
                value = value.replace("<image>", "").strip()
                if role == "human":
                    prompt_parts.append(value)
                elif role == "gpt":
                    response_parts.append(value)
            prompt = "\n".join(prompt_parts)
            response = "\n".join(response_parts)
        elif self.data_format == "spatial_qa":
            prompt = item.get("question", "")
            response = item.get("answer", "")
        else:
            raise ValueError(f"Unknown data format: {self.data_format}")

        return prompt, response

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.annotations[idx]

        # 加载图像
        image_file = item.get("image", "")
        image_path = os.path.join(self.data_root, "images", image_file)
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 构建对话
        prompt, response = self._format_conversation(item)

        # 构建 Qwen3.5 格式的 messages
        # 训练时需要包含 assistant 的回复
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response},
                ],
            },
        ]

        # 用 processor.apply_chat_template 生成输入
        # add_generation_prompt=False 因为已经包含了 assistant 回复
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        # 去掉 batch 维度（apply_chat_template 返回 batch_size=1）
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # 截断到 max_length
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # 构建 labels：只在 assistant 回复部分计算 loss
        labels = input_ids.clone()

        # 找到 assistant 回复的起始位置
        # Qwen3.5 chat template 中 assistant 回复前有特殊 token
        # 用 prompt-only 的 tokenization 来确定 prompt 长度
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_length = prompt_inputs["input_ids"].shape[1]
        prompt_length = min(prompt_length, labels.shape[0])

        # prompt 部分不计算 loss
        labels[:prompt_length] = IGNORE_INDEX
        # padding 部分不计算 loss
        labels[attention_mask == 0] = IGNORE_INDEX

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # 提取 pixel_values 和 image_grid_thw（由 processor 生成）
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)

        return result


def sft_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """SFT 数据集的 collate function（Qwen3.5 适配版）。

    处理变长的 input_ids（padding）和 pixel_values（concat）。
    """
    # 找到最大序列长度
    max_len = max(b["input_ids"].shape[0] for b in batch)
    batch_size = len(batch)

    # Padding input_ids, labels, attention_mask
    pad_token_id = 0  # Qwen3.5 的 pad_token_id
    padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    padded_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, sample in enumerate(batch):
        seq_len = sample["input_ids"].shape[0]
        padded_input_ids[i, :seq_len] = sample["input_ids"]
        padded_labels[i, :seq_len] = sample["labels"]
        padded_attention_mask[i, :seq_len] = sample["attention_mask"]

    result = {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": padded_attention_mask,
    }

    # pixel_values: 每个样本可能有不同数量的 patches，直接 concat
    if "pixel_values" in batch[0]:
        all_pixel_values = [b["pixel_values"] for b in batch]
        result["pixel_values"] = torch.cat(all_pixel_values, dim=0)

    # image_grid_thw: 每个样本一行 (T, H, W)，stack 成 (num_images, 3)
    if "image_grid_thw" in batch[0]:
        all_grid_thw = [b["image_grid_thw"] for b in batch]
        # 每个样本的 image_grid_thw 可能是 (1, 3) 或 (num_images, 3)
        result["image_grid_thw"] = torch.cat(
            [g.unsqueeze(0) if g.dim() == 1 else g for g in all_grid_thw], dim=0
        )

    return result