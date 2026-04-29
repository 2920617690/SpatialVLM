# 基于 Qwen3.5-4B 的 AVV

## 1. 具体底座

这套代码不再讨论抽象 VLM，而是直接假设一个真实的多模态底座：

```text
image -> ViT -> projector -> decoder-only LLM
```

默认目标模型是 `Qwen3.5-4B`。

这里的核心判断不是“要再加很多推理模块”，而是：

```text
同一个 VLM 应该学会 draft -> verify -> final 这个控制流
```

## 2. 三种监督模式

共享的 Qwen 模型会被训练成三种模式：

### Draft Mode

```text
输入: 图像 + 问题
输出: 结构化草案
```

草案里包含：

- 预测答案
- draft claim
- 分解后的 subclaims

### Verify Mode

```text
输入: 图像 + 问题 + draft claim
输出: 结构化验证结果
```

验证结果里包含：

- 每个 subclaim 的 verdict
- 总体 verdict: `support / contradict / insufficient`

### Final Mode

```text
输入: 图像 + 问题 + draft response + verification response
输出: 最终答案
```

## 3. 策略学习

在热启动之后，同一个共享模型还会被当作一个高层动作策略：

```text
PROPOSE
VERIFY
REVISE
ANSWER
ABSTAIN
```

这个策略真正要学的不是“视觉能力本身”，而是：

- 什么时候该验证
- 什么时候该改写
- 什么时候该停止

## 4. BVPO

仓库里实现的 stage-2 算法叫：

`BVPO = Budgeted Verification Policy Optimization`

奖励由四部分组成：

```text
R =
  answer_reward
  + verify_reward
  - step_cost
  - contradiction_penalty
```

核心思想是：验证应该有价值，但不能无限做。多看一步是有成本的。

## 5. 为什么先做合成数据

在一开始不应该直接上 noisy 的真实 QA，而应该先用合成空间场景，因为它能给出：

- 精确对象身份
- 精确 bbox
- 精确 subclaims
- 精确 verification label
- 精确 imitation trajectory

这对 `verify mode` 和 stage-1 imitation 尤其关键。

## 6. 样本结构

每个训练样本都会保存：

```text
image_path
question
answer
draft_response
verify_response
final_response
subclaims
trajectory
scene_objects
```

这套结构可以同时服务：

- stage-0 SFT
- stage-1 imitation
- stage-2 RL rollout

## 7. 建议的实验顺序

推荐先按这个顺序做：

1. synthetic 数据上的 one-shot baseline
2. `draft / verify / final` 的 stage-0 SFT
3. oracle trajectory 的 stage-1 imitation
4. stage-2 BVPO 微调
5. 再迁移到真实数据

这个顺序很重要。如果 stage-0 和 stage-1 都没有帮助，就不应该指望 RL 单独把问题救回来。
