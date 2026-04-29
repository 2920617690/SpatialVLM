from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.data import load_avv_samples
from src.data.prompts import (
    DRAFT_SYSTEM_PROMPT,
    FINAL_SYSTEM_PROMPT,
    POLICY_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    build_draft_prompt,
    build_final_prompt,
    build_verify_prompt,
)
from src.model import QwenAVVAgent

from .environment import SelfVerificationEnv


@dataclass
class RolloutStep:
    system_prompt: str
    user_prompt: str
    assistant_text: str
    reward: float


class BudgetedVerificationPolicyOptimizer:
    def __init__(self, agent: QwenAVVAgent, config: Dict[str, Any], output_dir: Path) -> None:
        self.agent = agent
        self.config = config
        self.output_dir = output_dir
        self.samples = load_avv_samples(config["data"]["train_manifest"])
        if config["data"].get("max_train_samples") is not None:
            self.samples = self.samples[: config["data"]["max_train_samples"]]
        self.optimizer = torch.optim.AdamW(self.agent.model.parameters(), lr=config["policy"]["learning_rate"])

    def _generate_mode_output(self, env: SelfVerificationEnv, mode: str) -> str:
        sample = env.sample
        if mode == "draft":
            return self.agent.generate_text(
                image_path=sample.image_path,
                system_prompt=DRAFT_SYSTEM_PROMPT,
                user_prompt=build_draft_prompt(sample),
                max_new_tokens=self.config["model"]["max_generation_tokens"]["draft"],
                temperature=self.config["policy"]["rollout_temperature"],
                top_p=self.config["policy"]["rollout_top_p"],
            )
        if mode == "verify":
            return self.agent.generate_text(
                image_path=sample.image_path,
                system_prompt=VERIFY_SYSTEM_PROMPT,
                user_prompt=build_verify_prompt(sample, draft_response=env.state.draft_response),
                max_new_tokens=self.config["model"]["max_generation_tokens"]["verify"],
                temperature=self.config["policy"]["rollout_temperature"],
                top_p=self.config["policy"]["rollout_top_p"],
            )
        if mode == "final":
            return self.agent.generate_text(
                image_path=sample.image_path,
                system_prompt=FINAL_SYSTEM_PROMPT,
                user_prompt=build_final_prompt(
                    sample,
                    draft_response=env.state.draft_response,
                    verify_response=env.state.verify_response,
                ),
                max_new_tokens=self.config["model"]["max_generation_tokens"]["final"],
                temperature=self.config["policy"]["rollout_temperature"],
                top_p=self.config["policy"]["rollout_top_p"],
            )
        raise ValueError(f"Unknown mode: {mode}")

    def _policy_action(self, env: SelfVerificationEnv) -> str:
        policy_text = self.agent.generate_text(
            image_path=env.sample.image_path,
            system_prompt=POLICY_SYSTEM_PROMPT,
            user_prompt=env.build_policy_prompt(),
            max_new_tokens=self.config["model"]["max_generation_tokens"]["policy"],
            temperature=self.config["policy"]["rollout_temperature"],
            top_p=self.config["policy"]["rollout_top_p"],
        )
        parsed = self.agent.parse_json_response(policy_text)
        return str(parsed.get("action", "ABSTAIN")).upper()

    def _rollout(self, sample) -> List[RolloutStep]:
        env = SelfVerificationEnv(sample, self.config)
        rollout_steps: List[RolloutStep] = []

        while not env.state.done:
            policy_prompt = env.build_policy_prompt()
            action_text = self.agent.generate_text(
                image_path=sample.image_path,
                system_prompt=POLICY_SYSTEM_PROMPT,
                user_prompt=policy_prompt,
                max_new_tokens=self.config["model"]["max_generation_tokens"]["policy"],
                temperature=self.config["policy"]["rollout_temperature"],
                top_p=self.config["policy"]["rollout_top_p"],
            )
            parsed = self.agent.parse_json_response(action_text)
            action = str(parsed.get("action", "ABSTAIN")).upper()

            generated_output = None
            if action == "PROPOSE":
                generated_output = self._generate_mode_output(env, "draft")
            elif action == "VERIFY":
                generated_output = self._generate_mode_output(env, "verify")
            elif action == "REVISE":
                generated_output = self._generate_mode_output(env, "draft")
            elif action == "ANSWER":
                generated_output = self._generate_mode_output(env, "final")

            _, reward, done = env.step(action, generated_output=generated_output)
            rollout_steps.append(
                RolloutStep(
                    system_prompt=POLICY_SYSTEM_PROMPT,
                    user_prompt=policy_prompt,
                    assistant_text=action_text,
                    reward=reward,
                )
            )
            if done:
                break
        return rollout_steps

    def _discounted_returns(self, rewards: List[float]) -> List[float]:
        returns: List[float] = []
        running = 0.0
        gamma = self.config["policy"]["discount"]
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()
        if self.config["policy"].get("normalize_returns", True) and returns:
            tensor = torch.tensor(returns, dtype=torch.float32)
            if tensor.std().item() > 0:
                tensor = (tensor - tensor.mean()) / (tensor.std() + 1.0e-6)
            returns = tensor.tolist()
        return returns

    def train(self) -> None:
        episodes_per_epoch = min(self.config["policy"]["episodes_per_epoch"], len(self.samples))
        random.shuffle(self.samples)
        selected = self.samples[:episodes_per_epoch]

        self.agent.model.train()
        for episode_index, sample in enumerate(selected):
            rollout = self._rollout(sample)
            returns = self._discounted_returns([step.reward for step in rollout])
            if not rollout:
                continue

            losses = []
            for step, advantage in zip(rollout, returns):
                logprob = self.agent.score_completion(
                    image_path=sample.image_path,
                    system_prompt=step.system_prompt,
                    user_prompt=step.user_prompt,
                    assistant_text=step.assistant_text,
                )
                losses.append(-logprob * advantage)

            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.config["policy"]["gradient_clip_norm"])
            self.optimizer.step()

            if episode_index % 10 == 0:
                print(
                    f"[bvpo] episode={episode_index} "
                    f"num_steps={len(rollout)} "
                    f"mean_reward={sum(step.reward for step in rollout):.4f} "
                    f"loss={loss.item():.4f}"
                )

        save_dir = self.output_dir / "checkpoint-final"
        save_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.agent.model, "save_pretrained"):
            self.agent.model.save_pretrained(str(save_dir))
        if hasattr(self.agent.processor, "save_pretrained"):
            self.agent.processor.save_pretrained(str(save_dir))
