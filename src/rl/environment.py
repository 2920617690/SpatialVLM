from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.data.prompts import (
    FINAL_SYSTEM_PROMPT,
    POLICY_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    build_final_prompt,
    build_policy_prompt,
    build_verify_prompt,
)
from src.data.schema import AVVSample


@dataclass
class EnvState:
    draft_response: Optional[str] = None
    verify_response: Optional[str] = None
    final_response: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    done: bool = False
    last_reward: float = 0.0


class SelfVerificationEnv:
    def __init__(self, sample: AVVSample, config: Dict[str, Any]) -> None:
        self.sample = sample
        self.config = config
        self.state = EnvState()
        self.max_steps = config["policy"]["max_episode_steps"]
        self.step_count = 0
        self.oracle_verify = json.loads(sample.verify_response)

    def build_policy_prompt(self) -> str:
        return build_policy_prompt(
            self.sample,
            history=self.state.history,
            draft_response=self.state.draft_response,
            verify_response=self.state.verify_response,
        )

    def _reward_verify(self, verify_response: str) -> float:
        reward = -self.config["policy"]["step_cost"]
        try:
            predicted = json.loads(verify_response)
            if predicted.get("overall_verdict") == self.oracle_verify.get("overall_verdict"):
                reward += self.config["policy"]["verify_reward"]
            else:
                reward -= self.config["policy"]["verify_reward"]
        except json.JSONDecodeError:
            reward -= self.config["policy"]["verify_reward"]
        return reward

    def _reward_answer(self, final_response: str) -> float:
        reward = 0.0
        try:
            predicted = json.loads(final_response)
            if str(predicted.get("final_answer", "")).strip().lower() == str(self.sample.answer).strip().lower():
                reward += self.config["policy"]["answer_reward"]
            else:
                reward -= self.config["policy"]["answer_reward"]
            if (
                self.state.verify_response is not None
                and self.oracle_verify.get("overall_verdict") == "contradict"
                and str(predicted.get("final_answer", "")).strip().lower() == "yes"
            ):
                reward -= self.config["policy"]["contradiction_penalty"]
        except json.JSONDecodeError:
            reward -= self.config["policy"]["answer_reward"]
        return reward

    def step(self, action: str, generated_output: Optional[str] = None) -> tuple[EnvState, float, bool]:
        self.step_count += 1
        reward = 0.0

        if action == "PROPOSE" and generated_output is not None:
            self.state.draft_response = generated_output
            self.state.history.append({"action": "PROPOSE", "observation": "draft created"})
        elif action == "VERIFY" and generated_output is not None:
            self.state.verify_response = generated_output
            reward += self._reward_verify(generated_output)
            self.state.history.append({"action": "VERIFY", "observation": "verification updated"})
        elif action == "REVISE" and generated_output is not None:
            self.state.draft_response = generated_output
            reward -= self.config["policy"]["step_cost"]
            self.state.history.append({"action": "REVISE", "observation": "draft revised"})
        elif action == "ANSWER" and generated_output is not None:
            self.state.final_response = generated_output
            reward += self._reward_answer(generated_output)
            self.state.history.append({"action": "ANSWER", "observation": "episode ended with answer"})
            self.state.done = True
        elif action == "ABSTAIN":
            reward += self.config["policy"]["abstain_reward"]
            self.state.history.append({"action": "ABSTAIN", "observation": "episode ended with abstain"})
            self.state.done = True
        else:
            reward -= self.config["policy"]["step_cost"]
            self.state.history.append({"action": action, "observation": "invalid or empty transition"})

        if self.step_count >= self.max_steps:
            self.state.done = True
        self.state.last_reward = reward
        return self.state, reward, self.state.done
