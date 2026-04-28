from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import SimpleBackboneAdapter
from .cropper import ROICropper
from .pointer import EvidencePointerHead
from .proposal import ProposalHead
from .verifier import RelationVerifier


@dataclass
class AVVOutput:
    answer_logits: torch.Tensor
    final_answer_logits: torch.Tensor
    evidence_query: torch.Tensor
    boxes_a: torch.Tensor
    boxes_b: torch.Tensor
    verifier_logits: torch.Tensor


class AVVModel(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 224,
        hidden_dim: int = 256,
        question_dim: int = 256,
        vocab_size: int = 4096,
        answer_classes: int = 2,
        verifier_classes: int = 3,
        crop_size: int = 160,
    ) -> None:
        super().__init__()
        self.backbone = SimpleBackboneAdapter(
            image_channels=image_channels,
            image_size=image_size,
            hidden_dim=hidden_dim,
            question_dim=question_dim,
            vocab_size=vocab_size,
        )
        self.proposal = ProposalHead(hidden_dim=hidden_dim, answer_classes=answer_classes)
        self.pointer = EvidencePointerHead(hidden_dim=hidden_dim)
        self.cropper = ROICropper(crop_size=crop_size)
        self.verifier = RelationVerifier(hidden_dim=hidden_dim, verifier_classes=verifier_classes)
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim + verifier_classes + answer_classes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, answer_classes),
        )

    def forward(self, images: torch.Tensor, question_tokens: torch.Tensor) -> AVVOutput:
        backbone = self.backbone(images, question_tokens)
        proposal = self.proposal(backbone.fused_context)
        pointer = self.pointer(
            patch_features=backbone.patch_features,
            question_features=backbone.question_features,
            evidence_query=proposal.evidence_query,
        )
        crop_a = self.cropper(images, pointer.boxes_a)
        crop_b = self.cropper(images, pointer.boxes_b)
        verifier_logits = self.verifier(
            crop_a=crop_a,
            crop_b=crop_b,
            claim_features=proposal.evidence_query,
        )
        verifier_probs = verifier_logits.softmax(dim=-1)
        proposal_probs = proposal.answer_logits.softmax(dim=-1)
        final_answer_logits = self.decision_head(
            torch.cat([backbone.fused_context, verifier_probs, proposal_probs], dim=-1)
        )
        return AVVOutput(
            answer_logits=proposal.answer_logits,
            final_answer_logits=final_answer_logits,
            evidence_query=proposal.evidence_query,
            boxes_a=pointer.boxes_a,
            boxes_b=pointer.boxes_b,
            verifier_logits=verifier_logits,
        )
