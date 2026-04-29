from __future__ import annotations

import json
from typing import Dict, List, Sequence

from .schema import AVVSample, TrajectoryStep


DRAFT_SYSTEM_PROMPT = (
    "You are a multimodal spatial reasoning model. "
    "Produce a strict JSON draft with keys: action, draft_answer, draft_claim, subclaims."
)

VERIFY_SYSTEM_PROMPT = (
    "You are a multimodal verification model. "
    "Given an image and a draft claim, return strict JSON with keys: action, overall_verdict, checked_claims."
)

FINAL_SYSTEM_PROMPT = (
    "You are a multimodal answer model. "
    "Given a question, a draft, and a verification result, return strict JSON with keys: action, final_answer, short_explanation."
)

POLICY_SYSTEM_PROMPT = (
    "You control a shared VLM policy. "
    "Choose the next action as strict JSON with a single key named action. "
    "Allowed values are PROPOSE, VERIFY, REVISE, ANSWER, ABSTAIN."
)


def _subclaim_lines(sample: AVVSample) -> str:
    if not sample.subclaims:
        return "- none"
    return "\n".join(f"- {item.claim_id}: {item.text}" for item in sample.subclaims)


def build_draft_prompt(sample: AVVSample) -> str:
    return (
        f"Question:\n{sample.question}\n\n"
        f"Known subclaims to reason about:\n{_subclaim_lines(sample)}\n\n"
        "Output a draft response JSON."
    )


def build_verify_prompt(sample: AVVSample, draft_response: str | None = None) -> str:
    draft = draft_response or sample.draft_response
    return (
        f"Question:\n{sample.question}\n\n"
        f"Draft response:\n{draft}\n\n"
        f"Subclaims to check:\n{_subclaim_lines(sample)}\n\n"
        "Check the image and return a verification JSON."
    )


def build_final_prompt(
    sample: AVVSample,
    draft_response: str | None = None,
    verify_response: str | None = None,
) -> str:
    draft = draft_response or sample.draft_response
    verify = verify_response or sample.verify_response
    return (
        f"Question:\n{sample.question}\n\n"
        f"Draft response:\n{draft}\n\n"
        f"Verification response:\n{verify}\n\n"
        "Return the final answer JSON."
    )


def build_policy_prompt(
    sample: AVVSample,
    history: Sequence[Dict[str, str]],
    draft_response: str | None = None,
    verify_response: str | None = None,
) -> str:
    history_text = "\n".join(
        f"- step {index + 1}: {item['action']} | observation: {item.get('observation', 'none')}"
        for index, item in enumerate(history)
    ) or "- no previous steps"
    draft = draft_response or "none"
    verify = verify_response or "none"
    return (
        f"Question:\n{sample.question}\n\n"
        f"Current draft response:\n{draft}\n\n"
        f"Current verification response:\n{verify}\n\n"
        f"History:\n{history_text}\n\n"
        "Choose the next action JSON."
    )


def mode_to_system_prompt(mode: str) -> str:
    if mode == "draft":
        return DRAFT_SYSTEM_PROMPT
    if mode == "verify":
        return VERIFY_SYSTEM_PROMPT
    if mode == "final":
        return FINAL_SYSTEM_PROMPT
    if mode == "policy":
        return POLICY_SYSTEM_PROMPT
    raise ValueError(f"Unknown mode: {mode}")


def mode_to_user_prompt(sample: AVVSample, mode: str) -> str:
    if mode == "draft":
        return build_draft_prompt(sample)
    if mode == "verify":
        return build_verify_prompt(sample)
    if mode == "final":
        return build_final_prompt(sample)
    raise ValueError(f"Unknown user prompt mode: {mode}")


def mode_to_target(sample: AVVSample, mode: str) -> str:
    if mode == "draft":
        return sample.draft_response
    if mode == "verify":
        return sample.verify_response
    if mode == "final":
        return sample.final_response
    raise ValueError(f"Unknown target mode: {mode}")


def trajectory_target(step: TrajectoryStep) -> str:
    return step.target_output


def action_to_json(action: str) -> str:
    return json.dumps({"action": action}, ensure_ascii=False)
