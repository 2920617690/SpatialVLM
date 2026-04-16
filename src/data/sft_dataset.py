"""
SFT Dataset — 指令微调数据集

支持两种格式：
1. LLaVA-style: {"id": ..., "image": "xxx.jpg", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
2. Spatial QA: {"image": "xxx.jpg", "question": "...", "answer": "..."}

用于阶段 1（对齐预训练）和阶段 2（全量微调）。
"""

import json
import os
import random
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# 特殊 token 占位符
IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100


class SFTDataset(Dataset):
    """指令微调数据集。

    每个样本返回：
    - pixel_values: (3, image_size, image_size)
    - input_ids: (seq_len,) tokenized 输入
    - labels: (seq_len,) 训练标签（human 部分为 IGNORE_INDEX）
    - attention_mask: (seq_len,)
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        tokenizer,
        image_size: int = 384,
        max_length: int = 2048,
        data_format: str = "llava",
    ):
        """
        Args:
            data_root: 数据根目录（包含 images/ 子目录）
            annotation_file: 标注文件路径（JSON）
            tokenizer: HuggingFace tokenizer
            image_size: 图像尺寸
            max_length: 文本最大 token 长度
            data_format: "llava" 或 "spatial_qa"
        """
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_format = data_format

        # 加载标注
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        print(f"加载 {len(self.annotations)} 条标注 from {annotation_file}")

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.annotations)

    def _format_conversation(self, item: Dict[str, Any]) -> str:
        """将标注转为统一的对话格式，返回 (prompt, response)。"""
        if self.data_format == "llava":
            conversations = item.get("conversations", [])
            prompt_parts = []
            response_parts = []
            for turn in conversations:
                role = turn.get("from", "")
                value = turn.get("value", "")
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
            pixel_values = self.transform(image)
        except (FileNotFoundError, OSError):
            # 损坏文件，随机替换
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 构建对话
        prompt, response = self._format_conversation(item)

        # Tokenize
        # 格式: [BOS] <image>\n{prompt} [/INST] {response} [EOS]
        prompt_text = f"{IMAGE_TOKEN}\n{prompt}"
        full_text = f"{prompt_text} {response}"

        prompt_encoding = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        input_ids = torch.tensor(full_encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full_encoding["attention_mask"], dtype=torch.long)

        # Labels: prompt 部分设为 IGNORE_INDEX，只在 response 部分计算 loss
        labels = input_ids.clone()
        prompt_length = len(prompt_encoding["input_ids"])
        labels[:prompt_length] = IGNORE_INDEX
        # padding 部分也设为 IGNORE_INDEX
        labels[attention_mask == 0] = IGNORE_INDEX

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt": prompt,
            "response": response,
        }


def sft_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """SFT 数据集的 collate function。"""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }
