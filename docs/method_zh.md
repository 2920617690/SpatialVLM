# QCR：问题条件下的二次视觉编码

## 1. 目标

这个项目现在只围绕一个更小的问题展开：

**如果模型在知道问题之后，再对同一张图做一次编码，空间推理会不会更强？**

所以这版方法刻意不先引入：

- crop 策略
- box 回归
- inspect controller
- RL 动作学习

我们先验证最小机制本身。

## 2. 两个模型

仓库现在只比较两个核心模型：

1. `one-pass baseline`
2. `QCR two-pass`

### Baseline

```text
image -> ViT -> projector -> decoder -> answer
```

### QCR

```text
第一遍:
  image -> ViT -> projector -> decoder
  读取 <reencode_slot> 的 hidden state

第二遍:
  把这个 hidden state 投影成一个 condition token
  prepend 到同一张图的 token 序列前面
  再做一次图像编码
  用残差方式修正第一次视觉表示
  decoder 输出最终答案
```

## 3. 前向过程

记：

- `P` 为图像 patch token
- `G1` 为第一次视觉 token
- `h_q` 为 `<reencode_slot>` 的 hidden state
- `c` 为 condition token
- `G2` 为第二次视觉 token
- `R` 为最终 refined visual token

### 第一遍

```text
P = PatchEmbed(x)
G1 = ViT(P)
h_q = Decoder(Project(G1), q + <reencode_slot>)
```

### condition token

```text
c = W_c(h_q)
```

### 第二遍

```text
G2_full = ViT([c ; P])
G2 = G2_full[:, 1:, :]
```

### 残差融合

```text
alpha = sigmoid(W_a(h_q))
R = G1 + alpha * (G2 - G1)
```

最后 decoder 看到的是 `Project(R)`，不是 `Project(G1)`。

## 4. 训练

当前合成数据提供：

- `question`
- `draft_response`
- `final_response`

baseline 只用 `final_response`。

QCR 使用：

- 可选的第一遍 draft loss
- 第二遍 final answer loss

```text
L = lambda_draft * L_draft + lambda_final * L_final
```

## 5. 为什么先做这版

QCR v0 的目标不是直接做一个复杂系统，而是先回答：

**question-conditioned second-pass encoding 到底有没有用？**

如果这一步没收益，那后面再加 crop、再加 inspect、再加 RL 都没有意义。

## 6. 严格后端与回退后端

理想实现是：

```text
[condition token ; image patch tokens] -> same ViT blocks
```

但不同 Qwen checkpoint 暴露的视觉塔内部结构可能不同。

所以代码保留两种 backend：

1. `strict_shared_vit`
2. `projected_token_refiner` fallback

fallback 仍然保留两遍编码逻辑，只是当拿不到视觉塔 block 接口时，在 projected visual token 空间里做近似重编码。

## 7. 最先要做的对照

最小实验表应该是：

1. one-pass baseline
2. two-pass + dummy condition token
3. two-pass + question-conditioned token

这样才能区分：

- 两遍编码本身有没有帮助
- 提升是不是只是多算一遍
- 还是 question-conditioned token 真正改变了视觉编码
