"""
预计算文本特征脚本

使用 SigLIP 的 text encoder 将 RefCOCO 的 referring expressions 编码为特征向量，
保存为 .pt 文件，避免训练时每个 batch 都跑一次 text encoder。

输出格式：
{
    "features": Tensor (N, max_len, text_dim),   # 文本特征
    "padding_masks": Tensor (N, max_len),         # padding mask (True=padding)
}
"""

import argparse
import json
import os
import pickle
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class TextListDataset(Dataset):
    """简单的文本列表 Dataset。"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 77):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def load_refcoco_texts(data_root: str) -> List[str]:
    """从 RefCOCO 数据中提取所有 referring expressions。"""
    texts = []

    # 查找 refs 文件
    refs_file = None
    for name in ["refs.p", "refs(unc).p", "refs(google).p"]:
        path = os.path.join(data_root, name)
        if os.path.exists(path):
            refs_file = path
            break

    if refs_file is None:
        raise FileNotFoundError(
            f"No refs file found in {data_root}. "
            "Expected: refs.p, refs(unc).p, or refs(google).p"
        )

    print(f"加载 refs 文件: {refs_file}")
    with open(refs_file, "rb") as f:
        refs = pickle.load(f)

    for ref in refs:
        for sent in ref.get("sentences", []):
            text = sent.get("raw", sent.get("sent", ""))
            if text:
                texts.append(text)

    print(f"共提取 {len(texts)} 条 referring expressions")
    return texts


def main():
    parser = argparse.ArgumentParser(description="预计算文本特征")
    parser.add_argument("--data_root", type=str, required=True, help="RefCOCO 数据根目录")
    parser.add_argument("--output", type=str, required=True, help="输出 .pt 文件路径")
    parser.add_argument("--model_name", type=str, default="google/siglip-so400m-patch14-384",
                        help="SigLIP 模型名称")
    parser.add_argument("--max_length", type=int, default=64, help="文本最大长度（SigLIP SO400M max_position_embeddings=64）")
    parser.add_argument("--batch_size", type=int, default=256, help="批大小")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cuda/cpu)")
    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载文本
    texts = load_refcoco_texts(args.data_root)

    # 加载 tokenizer 和 text encoder
    print(f"加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    text_encoder = model.text_model
    text_encoder = text_encoder.to(device).eval()

    # 获取 text_dim
    with torch.no_grad():
        dummy_input = tokenizer("test", return_tensors="pt", padding="max_length",
                                max_length=args.max_length, truncation=True)
        dummy_output = text_encoder(
            input_ids=dummy_input["input_ids"].to(device),
        )
        if hasattr(dummy_output, "last_hidden_state"):
            text_dim = dummy_output.last_hidden_state.shape[-1]
        else:
            text_dim = dummy_output[0].shape[-1]
    print(f"Text dim: {text_dim}")

    # 创建 DataLoader
    dataset = TextListDataset(texts, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 编码所有文本
    all_features = []
    all_padding_masks = []

    print("开始编码文本特征...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = text_encoder(input_ids=input_ids)
            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state
            else:
                features = outputs[0]

            # padding mask: attention_mask=0 的位置是 padding
            padding_mask = ~attention_mask.bool()

            all_features.append(features.cpu())
            all_padding_masks.append(padding_mask.cpu())

    # 拼接
    all_features = torch.cat(all_features, dim=0)       # (N, max_len, text_dim)
    all_padding_masks = torch.cat(all_padding_masks, dim=0)  # (N, max_len)

    print(f"特征形状: {all_features.shape}")
    print(f"Padding mask 形状: {all_padding_masks.shape}")

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "features": all_features,
        "padding_masks": all_padding_masks,
    }, args.output)
    print(f"已保存到: {args.output}")

    # 文件大小
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"文件大小: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
