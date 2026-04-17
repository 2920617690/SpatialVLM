"""
Spatial Pretrain Dataset — 预训练数据集

支持两种格式：
1. RefCOCO-style: image + referring expression + bbox（主要数据源）
2. Caption-style: image + caption，自动检测空间词

文本特征推荐预计算以避免每 batch 跑 LLaMA。
"""

import json
import os
import pickle
import random
from typing import Optional, List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from src.training.pretrain_spatial import SpatialMaskGenerator


# SigLIP 归一化参数
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# 左右互换词表（水平翻转增强时使用）
LR_SWAP_PAIRS = [
    ("left", "right"), ("左边", "右边"), ("左侧", "右侧"),
    ("左方", "右方"), ("左面", "右面"),
]


class SpatialPretrainDataset(Dataset):
    """Text-Conditioned Latent Prediction 预训练数据集。

    每个样本返回：
    - pixel_values: (3, 384, 384) 预处理后的图像
    - text_tokens: (T, text_dim) 文本特征（预计算或 dummy）
    - text_raw: str 原始文本
    - bbox: Optional[Tuple[float,float,float,float]] 归一化 bbox
    - text_padding_mask: (T,) bool padding mask
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        data_format: str = "refcoco",
        image_size: int = 384,
        max_text_length: int = 64,
        precomputed_text_features: Optional[str] = None,
        spatial_filter: bool = True,
        augment: bool = True,
    ):
        """
        Args:
            data_root: 数据根目录
            split: "train" / "val" / "test"
            data_format: "refcoco" 或 "caption"
            image_size: 图像尺寸
            max_text_length: 文本最大长度
            precomputed_text_features: 预计算文本特征 .pt 文件路径
            spatial_filter: 是否只保留含空间词的样本
            augment: 是否开启数据增强
        """
        self.data_root = data_root
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.augment = augment
        self.spatial_filter = spatial_filter

        # 加载样本列表
        if data_format == "refcoco":
            self.samples = self._load_refcoco(data_root, split)
        elif data_format == "caption":
            self.samples = self._load_caption(data_root, split)
        else:
            raise ValueError(f"Unknown data format: {data_format}")

        # 空间词过滤
        if spatial_filter:
            mask_gen = SpatialMaskGenerator()
            self.samples = [
                s for s in self.samples
                if mask_gen.detect_spatial_direction(s["text"]) is not None
                or s.get("bbox") is not None
            ]

        # 预计算文本特征
        self.text_features = None
        self.text_padding_masks = None
        if precomputed_text_features and os.path.exists(precomputed_text_features):
            data = torch.load(precomputed_text_features, map_location="cpu")
            self.text_features = data["features"]       # (N, T, text_dim)
            self.text_padding_masks = data["padding_masks"]  # (N, T)

        # 图像预处理
        self.transform = self._build_transform(augment)

    def _load_refcoco(self, data_root: str, split: str) -> List[Dict[str, Any]]:
        """加载 RefCOCO/RefCOCOg 数据。

        期望目录结构：
        data_root/
        ├── images/          # COCO images
        ├── instances.json   # COCO annotations
        └── refs.p or refs(unc).p  # referring expressions
        """
        # 加载 COCO annotations
        ann_file = os.path.join(data_root, "instances.json")
        if os.path.exists(ann_file):
            with open(ann_file) as f:
                coco = json.load(f)
            # 建立 annotation id → bbox/image_id 映射
            ann_map = {}
            for ann in coco["annotations"]:
                ann_map[ann["id"]] = {
                    "bbox": ann["bbox"],  # [x, y, w, h] 绝对坐标
                    "image_id": ann["image_id"],
                }
            img_map = {img["id"]: img for img in coco["images"]}
        else:
            ann_map, img_map = {}, {}

        # 加载 referring expressions
        refs_file = None
        for name in ["refs.p", "refs(unc).p", "refs(google).p"]:
            path = os.path.join(data_root, name)
            if os.path.exists(path):
                refs_file = path
                break

        samples = []
        if refs_file:
            with open(refs_file, "rb") as f:
                refs = pickle.load(f)

            for ref in refs:
                if ref.get("split", split) != split:
                    continue

                ann_id = ref["ann_id"]
                ann = ann_map.get(ann_id, {})
                image_id = ann.get("image_id", ref.get("image_id"))
                img_info = img_map.get(image_id, {})
                img_w = img_info.get("width", 1)
                img_h = img_info.get("height", 1)

                # bbox 归一化到 [0, 1]
                bbox = None
                if "bbox" in ann:
                    bx, by, bw, bh = ann["bbox"]
                    bbox = (bx / img_w, by / img_h, (bx + bw) / img_w, (by + bh) / img_h)

                # 图像路径
                file_name = img_info.get("file_name", f"COCO_train2014_{image_id:012d}.jpg")
                image_path = os.path.join(data_root, "images", file_name)

                # 每个 sentence 生成一个样本
                for sent in ref.get("sentences", []):
                    samples.append({
                        "image_path": image_path,
                        "text": sent.get("raw", sent.get("sent", "")),
                        "bbox": bbox,
                        "image_id": image_id,
                    })

        return samples

    def _load_caption(self, data_root: str, split: str) -> List[Dict[str, Any]]:
        """加载 image-caption 数据集（无 bbox）。

        期望目录结构：
        data_root/
        ├── images/
        └── captions_{split}.json   # [{"image": "xxx.jpg", "caption": "..."}]
        """
        caption_file = os.path.join(data_root, f"captions_{split}.json")
        samples = []
        if os.path.exists(caption_file):
            with open(caption_file) as f:
                data = json.load(f)
            for item in data:
                samples.append({
                    "image_path": os.path.join(data_root, "images", item["image"]),
                    "text": item["caption"],
                    "bbox": None,
                })
        return samples

    def _build_transform(self, augment: bool) -> transforms.Compose:
        """构建图像预处理 pipeline。"""
        if augment:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
            ])

    def _horizontal_flip(
        self, image: torch.Tensor, text: str, bbox: Optional[Tuple]
    ) -> Tuple[torch.Tensor, str, Optional[Tuple]]:
        """水平翻转增强：同时翻转图像、bbox、文本中的左右方向词。"""
        image = image.flip(-1)  # 水平翻转

        # 翻转 bbox
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox = (1 - x2, y1, 1 - x1, y2)

        # 交换文本中的左右方向词
        for left_word, right_word in LR_SWAP_PAIRS:
            placeholder = f"__SWAP_{left_word}__"
            text = text.replace(left_word, placeholder)
            text = text.replace(right_word, left_word)
            text = text.replace(placeholder, right_word)

        return image, text, bbox

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 加载图像
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            # 数据集可能有损坏文件，返回随机另一个样本
            return self.__getitem__(random.randint(0, len(self) - 1))

        pixel_values = self.transform(image)
        text = sample["text"]
        bbox = sample.get("bbox")

        # 随机水平翻转（50% 概率）
        if self.augment and random.random() < 0.5:
            pixel_values, text, bbox = self._horizontal_flip(pixel_values, text, bbox)

        # 文本特征
        if self.text_features is not None and idx < len(self.text_features):
            text_tokens = self.text_features[idx]
            text_padding_mask = self.text_padding_masks[idx]
        else:
            # Dummy（需要外部 text encoder on-the-fly 编码）
            text_tokens = torch.zeros(self.max_text_length, 4096)
            text_padding_mask = torch.ones(self.max_text_length, dtype=torch.bool)

        return {
            "pixel_values": pixel_values,
            "text_tokens": text_tokens,
            "text_raw": text,
            "bbox": bbox,
            "text_padding_mask": text_padding_mask,
        }


def spatial_pretrain_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function：处理可选 bbox 和变长文本。"""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "text_tokens": torch.stack([b["text_tokens"] for b in batch]),
        "texts": [b["text_raw"] for b in batch],
        "bboxes": [b["bbox"] for b in batch],
        "text_padding_mask": torch.stack([b["text_padding_mask"] for b in batch]),
    }
