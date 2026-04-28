# AVV: 面向 VLM 的自验证策略学习

## 1. 基本设定

我们讨论的仍然是标准 VLM 范式：

```text
image -> ViT -> projector -> decoder-only LLM -> answer
```

这个项目的核心判断是：很多空间错误并不只是因为模型能力不够，而是因为模型**过早提交答案**。它在真正验证视觉命题之前，就已经把答案说出来了。

因此，我们不优先重构整个架构，而是尽量保留原始的 `ViT + projector + decoder`，把问题从“一次性回答”改写成“多步自验证”。

## 2. 核心假设

对于空间问题，模型不应该只学会回答，还应该学会一个策略：

1. 什么时候先给出草案
2. 什么时候需要验证草案
3. 验证之后是否应该改写草案
4. 什么时候应该停止并提交最终答案

这件事更像顺序决策，而不是单步监督映射。

## 3. 为什么这不只是提示词工程

如果我们只是在推理时让同一个 VLM “多想一步”，那本质上只是 prompt engineering。

它要变成真正的方法，必须让模型在训练时学会三种不同的行为：

1. `draft mode`：生成草案答案或命题
2. `verify mode`：判断草案是否被图像支持
3. `final mode`：结合验证结果，生成最终答案

关键点不在于“提示词长什么样”，而在于共享参数的同一个模型是否被训练成会执行这三种模式。

## 4. 共享骨干的多步调用

我们刻意不把方法写成“必须加很多新模块”的形式。更理想的版本是复用同一个 VLM，分多次调用。

### 4.1 草案阶段

给定图像 `x` 和问题 `q`，decoder 先输出：

```text
y0 = draft answer
c0 = draft claim
```

例如：

```text
Question: Is the cat to the left of the dog?
Draft: answer = yes, claim = cat left of dog
```

### 4.2 验证阶段

然后再次调用同一个 VLM，进入 verify mode：

```text
input: image x + claim c0
output: v0 in {support, contradict, insufficient}
```

这里刻意把输出空间压得很小，目的就是减少模型自由发挥的 shortcut。

### 4.3 最终回答阶段

第三次调用模型：

```text
input: image x + question q + draft y0/c0 + verification result v0
output: final answer y*
```

最终答案可以保留、修改，或者拒绝初始草案。

## 5. RL 形式化

我们把自验证过程看成一个部分可观测的顺序决策问题。

### 5.1 状态

第 `t` 步的状态包含：

- 图像 `x`
- 问题 `q`
- 当前草案答案 `y_t`
- 当前命题 `c_t`
- 已有验证历史 `h_t`
- 剩余计算预算 `B_t`

这里的状态不是环境真状态，而是模型当前的信念和交互历史。

### 5.2 动作空间

第一版建议用一个小的离散动作空间：

```text
PROPOSE
VERIFY
REVISE
ANSWER
ABSTAIN
```

后续当然可以继续细化到区域级 reinspection，但初始版本更应该从高层控制动作开始。

### 5.3 状态转移

每个动作都触发同一个 VLM 在不同模式下再运行一次：

- `PROPOSE`：创建或更新草案命题
- `VERIFY`：检查当前命题是否被图像支持
- `REVISE`：在矛盾后重写命题或答案
- `ANSWER`：终止 episode 并输出最终答案
- `ABSTAIN`：终止 episode，但不冒险强答

### 5.4 奖励设计

一个实用的奖励函数应该同时考虑正确性和效率：

```text
R =
  final answer reward
  + verification consistency reward
  - unnecessary verification cost
  - contradiction-ignored penalty
```

最简单的一个版本可以是：

- 最终答案正确：`+1.0`
- 验证标签正确：`+0.2`
- 每多做一次验证：`-0.05`
- 明明出现矛盾还强行作答：`-0.3`
- 在真正不确定的情况下选择 abstain：给一个小正奖励

具体系数不是最关键的，原则是：
验证应当有价值，但无止境地检查也应当受到惩罚。

## 6. 训练流程

不建议一上来纯 RL。从零开始太难训，也太容易学歪。更稳的路线是三阶段。

### Stage 0: 监督式热启动

先让共享 VLM 学会三种格式：

#### Draft Mode

```text
input: image + question
target: draft answer + draft claim
```

#### Verify Mode

```text
input: image + claim
target: support / contradict / insufficient
```

#### Final Mode

```text
input: image + question + draft + verification
target: final answer
```

损失可以写成：

```text
L_sft =
  L_draft
  + lambda_verify * L_verify
  + lambda_final * L_final
```

### Stage 1: Oracle 引导的模仿学习

利用合成数据或 bbox 派生关系构造比较好的轨迹：

- 如果关系很明显，直接回答
- 如果关系困难或是反事实问题，先验证
- 如果验证与草案矛盾，就改写
- 再输出最终答案

这一阶段的作用，是先教会策略一个“能工作的控制流”。

### Stage 2: RL 微调

最后再做策略优化。

此时优化目标不再只是 token 模仿，而是期望回报：

```text
J(pi) = E_pi [ sum_t gamma^t r_t ]
```

具体算法可以灵活选择。PPO 风格的微调是比较自然的起点，但方法本身不依赖某一个特定优化器。

## 7. 数据构造

热启动和 imitation 阶段依然需要结构化监督。

每个样本应该从：

```text
(image, question, answer)
```

提升为：

```text
(image, question, claim, verify_label, final_answer)
```

主要数据来源：

- COCO instances
- Visual Genome
- RefCOCO / RefCOCOg
- CLEVR
- 合成 grid、chart、table、GUI 数据

对于有 bbox 的数据，可以程序化地推导关系标签：

- `left_of`
- `right_of`
- `above`
- `below`
- `overlap`
- `inside`
- `larger_than`
- `smaller_than`

这些数据尤其适合训练 verify mode，也适合构造 imitation trajectory。

## 8. 为什么 RL 更合适

监督学习擅长教会模型：

- 什么样的草案是合法的
- 什么样的验证标签是合法的
- 怎样把最终结论说成自然语言

RL 更适合解决监督不擅长的部分：

- 什么时候必须验证
- 应该验证多少次
- 什么时候该改写
- 什么时候应该停止

所以 RL 不该替代整个 VLM 训练，而应该聚焦在**自验证策略学习**上。

## 9. 最大风险

如果验证结果也完全由模型自己说了算，就很容易出现 reward hacking。

模型可能学会：

- 先编一个草案
- 再编一个和草案自洽的 verification label
- 最后在语言上自圆其说，却不是真的视觉忠实

因此，早期训练和奖励设计应尽量依赖外部真值：

- 合成几何任务
- bbox 派生关系
- 确定性的翻转一致性
- 程序化 relation checker

## 10. 优先做的消融

最先该做的消融包括：

1. 只做一次性回答
2. 只有 draft + final，没有 verify mode
3. 多步监督，但没有 RL
4. RL 中不惩罚额外验证步数
5. RL 中不惩罚忽略矛盾
6. 使用 GT relation label 与 noisy relation label 对比

这些消融能回答一个核心问题：
性能提升究竟来自真正的策略学习，还是仅仅来自更长的 prompting。
