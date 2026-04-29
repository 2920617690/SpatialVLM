from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
import torch

from src.model.chat import build_messages, render_chat_text


@dataclass
class QwenComponents:
    model: Any
    processor: Any


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_qwen_components(config: Dict[str, Any]) -> QwenComponents:
    from transformers import AutoProcessor

    model_cfg = config["model"]
    processor = AutoProcessor.from_pretrained(
        model_cfg["base_model_id"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    model = None
    model_kwargs = {
        "torch_dtype": _resolve_dtype(model_cfg.get("torch_dtype", "bfloat16")),
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
    }
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_cfg["base_model_id"], **model_kwargs)
    except Exception:
        pass

    if model is None:
        try:
            from transformers import AutoModelForVision2Seq

            model = AutoModelForVision2Seq.from_pretrained(model_cfg["base_model_id"], **model_kwargs)
        except Exception:
            pass

    if model is None:
        try:
            from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_cfg["base_model_id"], **model_kwargs)
        except Exception:
            pass

    if model is None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model_id"], **model_kwargs)

    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if config.get("lora", {}).get("enabled", False):
        from peft import LoraConfig, get_peft_model

        lora_cfg = config["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            bias=lora_cfg["bias"],
            target_modules=lora_cfg["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    if model_cfg.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return QwenComponents(model=model, processor=processor)


class QwenAVVAgent:
    def __init__(self, model: Any, processor: Any, config: Dict[str, Any]) -> None:
        self.model = model
        self.processor = processor
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QwenAVVAgent":
        components = load_qwen_components(config)
        return cls(components.model, components.processor, config)

    @staticmethod
    def _load_image(image_path: str | Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _prepare_inputs(
        self,
        image_path: str | Path,
        system_prompt: str,
        user_prompt: str,
        assistant_text: Optional[str] = None,
        add_generation_prompt: bool = False,
    ) -> Dict[str, Any]:
        image = self._load_image(image_path)
        messages = build_messages(system_prompt, user_prompt, assistant_text=assistant_text)
        chat_text = render_chat_text(self.processor, messages, add_generation_prompt=add_generation_prompt)
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.config["model"]["max_prompt_length"],
        )
        return inputs

    def generate_text(
        self,
        image_path: str | Path,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        inputs = self._prepare_inputs(
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_text=None,
            add_generation_prompt=True,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1.0e-5),
                top_p=top_p,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[0, prompt_len:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_json(
        self,
        image_path: str | Path,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict[str, Any]:
        text = self.generate_text(
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return self.parse_json_response(text)

    @staticmethod
    def parse_json_response(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
        return {"raw_text": text}

    def score_completion(
        self,
        image_path: str | Path,
        system_prompt: str,
        user_prompt: str,
        assistant_text: str,
    ) -> torch.Tensor:
        prompt_inputs = self._prepare_inputs(
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_text=None,
            add_generation_prompt=True,
        )
        full_inputs = self._prepare_inputs(
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_text=assistant_text,
            add_generation_prompt=False,
        )
        device = next(self.model.parameters()).device
        prompt_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in prompt_inputs.items()}
        full_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in full_inputs.items()}

        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        if "attention_mask" in full_inputs:
            labels[full_inputs["attention_mask"] == 0] = -100

        outputs = self.model(**full_inputs, labels=labels)
        token_count = (labels != -100).sum().clamp(min=1)
        return -outputs.loss * token_count
