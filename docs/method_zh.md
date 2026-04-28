# AVV: 回答, 视觉证据, 验证

## 1. 问题定义

给定图像 `x` 和问题 `q`，标准 VLM 通常直接建模：

```text
p(y | x, q)
```

其中 `y` 是答案文本。这个目标对空间推理来说约束太弱。它只奖励模型答对，却不要求模型保留、读取或核对支撑答案的视觉证据。

我们把空间问答改写为一个“命题验证”问题。模型先提出一个初始答案 `y0`，再收集局部证据，判断这些证据是否真的支持该答案隐含的空间命题，最后输出最终答案 `y*`。

## 2. 核心假设

空间错误往往来自于模型过早提交答案：

- 绑定错了实例
- 读错了局部区域
- 把错误的视觉细节压缩成了语言
- 从来没有回头检查自己的结论

因此，最重要的干预不是“继续直接回答”，而是：

**先回答，再基于视觉证据做一次验证。**

## 3. 总体流程

AVV 有五个阶段：

1. `Proposal`：先产生初始答案 `y0`
2. `Evidence Query`：生成一个紧凑的证据查询 `z0`，表示“如果这个答案是对的，我应该看什么”
3. `Evidence Localization`：定位对象 `A` 和对象 `B` 的证据区域
4. `Re-Perception`：从原图中裁出这些区域，并以更高保真度重新编码
5. `Verification`：判断重新读取到的证据是支持、反驳，还是不足以判断

最后由决策头综合 proposal 置信度和 verification 结果，输出最终答案 `y*`。

## 4. 模型结构

### 4.1 Backbone

backbone 负责编码：

- 图像 token `V = {v_i} for i = 1...N`
- 问题表示 `q_h`

这里不强绑定某个具体的 VLM 家族，只要求我们能拿到图像特征和问题条件向量。

### 4.2 Proposal Head

proposal head 输出：

```text
p_theta(y0 | x, q)
```

同时输出一个证据查询向量：

```text
z0 = f_evidence(V, q_h, y0)
```

`z0` 不是完整的 chain-of-thought，而是一个紧凑的 latent，表示模型认为下一步该验证什么。

### 4.3 Evidence Pointer

pointer head 预测证据区域：

```text
bA, bB = f_ptr(V, q_h, z0)
```

其中 `bA`、`bB` 可以是 box，也可以是 heatmap，对应这道空间题最关键的两个对象区域。

### 4.4 Re-Perception Module

AVV 和单纯 latent reasoning 最大的区别是：它会回到像素。

```text
xA_hat = crop(x, bA)
xB_hat = crop(x, bB)
```

然后重新编码，得到证据特征：

```text
fA = g(xA_hat)
fB = g(xB_hat)
```

这一步是为了纠正第一次粗读时的 patch 边界误差、局部读错和实例绑定错误。

### 4.5 Hypothesis Builder

将 `(q, y0)` 组合成一个结构化命题：

```text
c = h(q, y0)
```

例如：

- `Is the cat left of the dog?` + `yes` -> `cat left of dog`
- `Is the apple farther than the banana from the cup?` + `no` -> `apple is not farther than banana from cup`

### 4.6 Relation Verifier

verifier 输出：

```text
p_theta(v | fA, fB, c)
```

其中：

```text
v in {support, contradict, insufficient}
```

它不是生成式模块，而是一个判别器。它只回答一个很窄的问题：局部证据到底支不支持当前命题。

### 4.7 Final Decision Head

最终答案由 proposal 和 verifier 共同决定：

```text
p_theta(y* | y0, v, q_h)
```

最简单的策略是：

- 如果 `v = support`，保留 `y0`
- 如果 `v = contradict`，翻转或改写 `y0`
- 如果 `v = insufficient`，保守输出或触发回退策略

## 5. 训练目标

总损失写成：

```text
L =
  L_ans
  + lambda_final * L_final
  + lambda_ptr * L_ptr
  + lambda_ver * L_ver
  + lambda_cf * L_cf
  + lambda_cons * L_cons
```

### 5.1 Proposal Loss

```text
L_ans = CE(p_theta(y0 | x, q), y)
```

### 5.2 Final Answer Loss

```text
L_final = CE(p_theta(y* | x, q), y)
```

### 5.3 Pointer Loss

如果证据监督是 box：

```text
L_ptr =
  L1(bA, bA*)
  + L1(bB, bB*)
  + GIoU(bA, bA*)
  + GIoU(bB, bB*)
```

如果证据监督是 heatmap，也可以用 BCE 替代。

### 5.4 Verifier Loss

```text
L_ver = CE(p_theta(v | fA, fB, c), v*)
```

### 5.5 Counterfactual Loss

对同一对证据区域，构造一个错误命题 `c_tilde`，并要求 verifier 明确反驳：

```text
L_cf = -log p_theta(v = contradict | fA, fB, c_tilde)
```

这能减少模型仅凭对象共现关系走 shortcut。

### 5.6 Consistency Loss

对于几何变换 `g` 及其语言映射 `T_g`，要求验证结果满足一致性：

```text
L_cons =
  D(
    p_theta(v | g(x), T_g(q)),
    T_g(p_theta(v | x, q))
  )
```

最先做的就是水平翻转，以及 `left <-> right` 的同步替换。

## 6. 训练样本构造

每个样本不再只是 `(image, question, answer)`，而是：

```text
(x, A, B, r, q, y, bA*, bB*)
```

其中：

- `A, B` 是对象实例
- `r` 是几何关系
- `bA*, bB*` 是证据框

### 6.1 数据来源

优先使用：

- COCO instances
- Visual Genome
- RefCOCO / RefCOCOg
- CLEVR 或合成关系数据

### 6.2 关系标签

第一版关系集合建议限制为：

- `left_of`
- `right_of`
- `above`
- `below`
- `overlap`
- `inside`
- `intersect`
- `larger_than`
- `smaller_than`

距离和深度关系先放后面，等训练闭环稳定以后再加入。

### 6.3 正负样本构造

对每个有效对象对 `(A, B)`：

- 正样本问题：`Is A left of B?`
- 负样本问题：`Is A right of B?`

难负样本可以来自：

- 交换邻近的同类实例
- 替换成容易混淆的关系
- 故意破坏一个 crop，制造 `insufficient`

## 7. 分阶段训练

### Stage 1: Pointer Pretraining

第一阶段只训练证据定位：

```text
L_stage1 = L_ptr
```

### Stage 2: Verifier Pretraining

第二阶段用 GT crop 训练 verifier：

```text
L_stage2 =
  L_ver
  + lambda_cf * L_cf
  + lambda_cons * L_cons
```

### Stage 3: Joint Fine-tuning

第三阶段把 proposal、pointer、re-perception 和 verifier 串起来：

```text
L_stage3 =
  L_ans
  + lambda_final * L_final
  + lambda_ptr * L_ptr
  + lambda_ver * L_ver
  + lambda_cf * L_cf
  + lambda_cons * L_cons
```

## 8. 这条路线的区别

AVV 不默认假设问题在于“latent 迭代不够深”。它假设更直接的失败模式：

- 模型在真正检查关键视觉细节之前，就已经把答案说出口了

因此，它比重构整个 encoder-decoder 主干更小，但又比单纯加 reasoning token 更针对真实错误来源。

## 9. 建议优先做的消融

最关键的消融有这几类：

1. 去掉 re-perception，只在第一次前向的特征上做 verifier
2. 去掉 verifier，直接输出 proposal
3. 去掉 counterfactual training
4. 去掉 consistency loss
5. 使用 GT evidence box 与 predicted evidence box 对比

这些消融能回答一个关键问题：提升究竟来自真正的视觉验证，还是只是来自多加了参数。
