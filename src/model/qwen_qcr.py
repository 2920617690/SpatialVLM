from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.prompts import build_qcr_final_user_prompt, build_qcr_pass1_prompt


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
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model_id"], **model_kwargs)

    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return QwenComponents(model=model, processor=processor)


class QCRConditioner(nn.Module):
    def __init__(self, llm_dim: int, token_dim: int, gate_mode: str = "channel") -> None:
        super().__init__()
        self.to_condition = nn.Linear(llm_dim, token_dim)
        gate_dim = token_dim if gate_mode == "channel" else 1
        self.to_gate = nn.Linear(llm_dim, gate_dim)

    def forward(self, h_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition_token = self.to_condition(h_q).unsqueeze(1)
        alpha = torch.sigmoid(self.to_gate(h_q)).unsqueeze(1)
        return condition_token, alpha


class ProjectedTokenRefiner(nn.Module):
    def __init__(self, token_dim: int, num_layers: int = 2, num_heads: int = 8) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, visual_tokens: torch.Tensor, condition_token: torch.Tensor) -> torch.Tensor:
        refined = self.encoder(torch.cat([condition_token, visual_tokens], dim=1))
        return refined[:, 1:, :]


class VisionTowerAdapter:
    def __init__(self, base_model: Any) -> None:
        self.base_model = base_model
        self._warned_fallback = False

    def extract_visual_tokens(self, visual_inputs: Dict[str, Any]) -> torch.Tensor:
        candidate_methods = [
            getattr(self.base_model, "get_image_features", None),
            getattr(getattr(self.base_model, "model", None), "get_image_features", None),
        ]
        for method in candidate_methods:
            if method is None:
                continue
            try:
                features = method(**visual_inputs)
                if isinstance(features, tuple):
                    features = features[0]
                return features
            except TypeError:
                continue
        raise RuntimeError("Could not extract visual tokens from the loaded Qwen checkpoint.")

    def strict_reencode(self, visual_tokens: torch.Tensor, condition_token: torch.Tensor) -> Optional[torch.Tensor]:
        del visual_tokens, condition_token
        return None

    def warn_fallback(self) -> None:
        if not self._warned_fallback:
            warnings.warn(
                "Falling back to projected-token QCR because the loaded Qwen checkpoint "
                "does not expose a strict shared-ViT re-entry path.",
                stacklevel=2,
            )
            self._warned_fallback = True


class _BaseQwenWrapper(nn.Module):
    def __init__(self, base_model: Any, processor: Any, config: Dict[str, Any]) -> None:
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.config = config
        self.tokenizer = processor.tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    def _load_image(self, image_path: str | Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _prepare_visual_inputs(self, image: Image.Image) -> Dict[str, Any]:
        visual_inputs = self.processor(images=[image], return_tensors="pt")
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in visual_inputs.items()
        }

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(text, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    @staticmethod
    def _find_language_model(model: Any) -> Any:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model"):
            return model.model
        return model

    def _input_embeddings(self) -> Any:
        language_model = self._find_language_model(self.base_model)
        if hasattr(language_model, "get_input_embeddings"):
            return language_model.get_input_embeddings()
        return language_model.embed_tokens

    def _lm_head(self) -> Any:
        if hasattr(self.base_model, "lm_head"):
            return self.base_model.lm_head
        language_model = self._find_language_model(self.base_model)
        if hasattr(language_model, "lm_head"):
            return language_model.lm_head
        raise RuntimeError("Could not find lm_head on the loaded Qwen checkpoint.")

    def _language_forward(
        self,
        visual_tokens: torch.Tensor,
        text: str,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        text_tokens = self._tokenize_text(text)
        token_embeddings = self._input_embeddings()(text_tokens["input_ids"])
        inputs_embeds = torch.cat([visual_tokens, token_embeddings], dim=1)
        visual_mask = torch.ones(visual_tokens.shape[:2], dtype=text_tokens["attention_mask"].dtype, device=self.device)
        attention_mask = torch.cat([visual_mask, text_tokens["attention_mask"]], dim=1)
        language_model = self._find_language_model(self.base_model)
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        return {
            "outputs": outputs,
            "text_tokens": text_tokens,
            "visual_length": visual_tokens.shape[1],
        }

    def _causal_loss(
        self,
        visual_tokens: torch.Tensor,
        prompt_text: str,
        target_text: str,
    ) -> torch.Tensor:
        prompt_tokens = self._tokenize_text(prompt_text)
        full_tokens = self._tokenize_text(prompt_text + target_text)
        full_embeds = self._input_embeddings()(full_tokens["input_ids"])

        inputs_embeds = torch.cat([visual_tokens, full_embeds], dim=1)
        visual_mask = torch.ones(visual_tokens.shape[:2], dtype=full_tokens["attention_mask"].dtype, device=self.device)
        attention_mask = torch.cat([visual_mask, full_tokens["attention_mask"]], dim=1)
        labels = full_tokens["input_ids"].clone()
        prompt_len = prompt_tokens["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        labels = torch.cat(
            [
                torch.full((labels.shape[0], visual_tokens.shape[1]), -100, dtype=labels.dtype, device=self.device),
                labels,
            ],
            dim=1,
        )

        language_model = self._find_language_model(self.base_model)
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        logits = self._lm_head()(outputs.last_hidden_state)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


class QwenOnePassModel(_BaseQwenWrapper):
    def __init__(self, base_model: Any, processor: Any, config: Dict[str, Any]) -> None:
        super().__init__(base_model, processor, config)
        self.vision_adapter = VisionTowerAdapter(base_model)

    def forward_sample(self, sample: Any) -> torch.Tensor:
        image = self._load_image(sample.image_path)
        visual_inputs = self._prepare_visual_inputs(image)
        visual_tokens = self.vision_adapter.extract_visual_tokens(visual_inputs)
        prompt_text = f"Question:\n{sample.question}\n\nReturn the final answer JSON.\n"
        return self._causal_loss(visual_tokens, prompt_text=prompt_text, target_text=sample.final_response)


class QCRQwenModel(_BaseQwenWrapper):
    def __init__(self, base_model: Any, processor: Any, config: Dict[str, Any]) -> None:
        super().__init__(base_model, processor, config)
        llm_dim = config["model"]["llm_hidden_size"]
        self.vision_adapter = VisionTowerAdapter(base_model)
        self.conditioner = QCRConditioner(llm_dim=llm_dim, token_dim=llm_dim, gate_mode=config["model"]["residual_gate"])
        self.refiner = ProjectedTokenRefiner(token_dim=llm_dim, num_layers=2, num_heads=8)
        self.reencode_slot_token = config["model"]["reencode_slot_token"]
        self._maybe_add_reencode_slot()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QCRQwenModel":
        components = load_qwen_components(config)
        return cls(components.model, components.processor, config)

    def _maybe_add_reencode_slot(self) -> None:
        if self.reencode_slot_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.reencode_slot_token]})
            if hasattr(self.base_model, "resize_token_embeddings"):
                self.base_model.resize_token_embeddings(len(self.tokenizer))

    def _strict_or_fallback_reencode(self, visual_tokens: torch.Tensor, condition_token: torch.Tensor) -> torch.Tensor:
        backend = self.config["model"]["reencode_backend"]
        if backend == "strict_shared_vit":
            strict_tokens = self.vision_adapter.strict_reencode(visual_tokens, condition_token)
            if strict_tokens is not None:
                return strict_tokens
            self.vision_adapter.warn_fallback()
        return self.refiner(visual_tokens, condition_token)

    def _first_pass_state(self, visual_tokens: torch.Tensor, sample: Any) -> torch.Tensor:
        prompt = build_qcr_pass1_prompt(sample, self.reencode_slot_token) + f"\n{self.reencode_slot_token}"
        result = self._language_forward(visual_tokens, prompt, output_hidden_states=True)
        input_ids = result["text_tokens"]["input_ids"][0]
        slot_id = self.tokenizer.convert_tokens_to_ids(self.reencode_slot_token)
        matches = (input_ids == slot_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            raise RuntimeError("Could not find <reencode_slot> in the first-pass prompt.")
        text_position = matches[-1].item()
        seq_position = result["visual_length"] + text_position
        return result["outputs"].hidden_states[-1][:, seq_position, :]

    def forward_sample(self, sample: Any) -> Dict[str, torch.Tensor]:
        image = self._load_image(sample.image_path)
        visual_inputs = self._prepare_visual_inputs(image)
        first_pass_visual = self.vision_adapter.extract_visual_tokens(visual_inputs)
        h_q = self._first_pass_state(first_pass_visual, sample)
        condition_token, alpha = self.conditioner(h_q)
        second_pass_visual = self._strict_or_fallback_reencode(first_pass_visual, condition_token)
        refined_visual = first_pass_visual + alpha * (second_pass_visual - first_pass_visual)

        losses: Dict[str, torch.Tensor] = {}
        if self.config["loss"].get("use_draft_loss", True):
            draft_prompt = f"Question:\n{sample.question}\n\nProduce the draft response JSON.\n"
            losses["draft_loss"] = self._causal_loss(first_pass_visual, prompt_text=draft_prompt, target_text=sample.draft_response)
        final_prompt = build_qcr_final_user_prompt(sample) + "\n"
        losses["final_loss"] = self._causal_loss(refined_visual, prompt_text=final_prompt, target_text=sample.final_response)
        return losses
