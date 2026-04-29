from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_messages(system_prompt: str, user_prompt: str, assistant_text: Optional[str] = None) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    if assistant_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )
    return messages


def render_chat_text(processor: Any, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    fallback_lines = []
    for message in messages:
        parts = []
        for content in message["content"]:
            if content["type"] == "image":
                parts.append("<image>")
            else:
                parts.append(content["text"])
        fallback_lines.append(f"{message['role'].upper()}: {' '.join(parts)}")
    if add_generation_prompt:
        fallback_lines.append("ASSISTANT:")
    return "\n".join(fallback_lines)
