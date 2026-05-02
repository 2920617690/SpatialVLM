from __future__ import annotations

from .schema import AVVSample


BASELINE_SYSTEM_PROMPT = (
    "You are a multimodal spatial reasoning model. "
    "Answer the visual question with a strict JSON object containing keys: final_answer, short_explanation."
)

DRAFT_SYSTEM_PROMPT = (
    "You are a multimodal spatial reasoning model. "
    "Produce a strict JSON draft with keys: draft_answer, draft_claim, subclaims."
)

FINAL_SYSTEM_PROMPT = (
    "You are a multimodal spatial reasoning model. "
    "Produce a strict JSON final answer with keys: final_answer, short_explanation."
)


def build_baseline_user_prompt(sample: AVVSample) -> str:
    return f"Question:\n{sample.question}\n\nReturn the final answer JSON."


def build_draft_user_prompt(sample: AVVSample) -> str:
    return (
        f"Question:\n{sample.question}\n\n"
        "Produce a draft JSON with a draft answer, one draft claim, and the relevant subclaims."
    )


def build_qcr_pass1_prompt(sample: AVVSample, reencode_slot_token: str) -> str:
    return (
        f"Question:\n{sample.question}\n\n"
        f"Route the question-conditioned visual state through {reencode_slot_token}."
    )


def build_qcr_final_user_prompt(sample: AVVSample) -> str:
    return f"Question:\n{sample.question}\n\nReturn the final answer JSON."


def mode_to_system_prompt(mode: str) -> str:
    if mode == "baseline":
        return BASELINE_SYSTEM_PROMPT
    if mode == "draft":
        return DRAFT_SYSTEM_PROMPT
    if mode == "final":
        return FINAL_SYSTEM_PROMPT
    raise ValueError(f"Unknown mode: {mode}")


def mode_to_user_prompt(sample: AVVSample, mode: str) -> str:
    if mode == "baseline":
        return build_baseline_user_prompt(sample)
    if mode == "draft":
        return build_draft_user_prompt(sample)
    if mode == "final":
        return build_qcr_final_user_prompt(sample)
    raise ValueError(f"Unknown user prompt mode: {mode}")


def mode_to_target(sample: AVVSample, mode: str) -> str:
    if mode == "baseline":
        return sample.final_response
    if mode == "draft":
        return sample.draft_response
    if mode == "final":
        return sample.final_response
    raise ValueError(f"Unknown target mode: {mode}")
