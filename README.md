# Spatial VLM：增强视觉语言模型的空间推理能力

## 研究动机

当前主流 VLM（如 LLaVA、Qwen-VL、InternVL）在空间推理任务上表现不佳。

### 问题分析

空间信息在 VLM pipeline 的三个环节被系统性丢失：

1. **ViT 编码阶段**：ViT 编码时完全不知道下游要问什么问题，对所有 patch 一视同仁地编码，空间信息在 flatten 过程中被稀释
2. **Projector 阶段**：逐 patch 独立投影（MLP），patch 间的空间关系未被显式编码
3. **LLM 解码阶段**：1D RoPE 只编码序列位置，无法区分"水平相邻"和"垂直相邻"的 patch；cross attention 在 attend 视觉特征时不考虑空间位置

### 现有方法的局限

- **Spatial-SSRL (2025)**：设计 5 个空间预训练任务，平均仅提升 4.63%
- **Spa3R (2025)**：多视角学习空间表征，VSI-Bench 上 58.6%
- **V-JEPA 2 (2025)**：100 万小时视频自监督预训练 + 4.7 倍数据扩增，在 Video QA 上也仅涨几个点（PerceptionTest +1.3, TempCompass +4.2）

这些方法都在试图通过 **更多/更好的训练信号** 来让现有架构学到空间理解，但效果有限，说明 **"更多训练信号"路线的边际收益在递减**。问题不在训练信号，而在架构本身。

### 核心洞察

人类看图像是 **"带着意图去看"** 的 —— 是 attention-driven 的主动感知。当问题是"桌子左边是什么？"时，人会主动将注意力聚焦到图像左侧区域。而现有 VLM 的 ViT 在编码时完全不知道要关注什么，是被动的、无差别的编码。

参考 **ViLT** 将文本和图像一起输入做联合编码的思路，结合对 VLM pipeline 各环节的针对性改进，本项目从 **架构层面** 提出三个互补的机制。

## 核心方案

### 方案 A：DFlash 风格 KV Injection（Text-Conditioned ViT）

**核心思路**：受 DFlash（2602.06036）启发，将文本特征直接注入 ViT 每层 attention 的 KV cache，让 visual patch queries 直接 attend 到文本 KV entries，实现持续、不稀释的文本引导。

**"第三人称"定位**：文本块不独立理解图像，只作为引导器持续提供空间意图条件——类比 DFlash 中扩散块借助 target model 的深层语义做高效并行 draft。

- **KV Injection**：将文本 embeddings 压缩为 compact context feature (256d)，通过每层独立的 K/V 投影头注入 ViT attention
- 每层 attention 的 K/V 被扩展：`K = [K_visual | K_text]`，`V = [V_visual | V_text]`
- **全层注入**（默认 27 层全部注入），文本引导信号在每一层都持续存在，不随层数增加而稀释
- Per-head gate（初始化为 0），启动时等价于原始 ViT，渐进式引入文本条件
- 通过 monkey-patch SigLIP attention forward 实现，对 ViT 其他部分完全透明

**与 DFlash 的类比**：

| DFlash | 方案 A |
|--------|--------|
| target LLM hidden features → fuse → inject into drafter KV | text encoder features → compress → inject into ViT KV |
| 扩散块作为 decoder 的轻量 adapter | 文本块作为 ViT 的轻量 adapter |
| KV injection 到每层，信号不稀释 | KV injection 到 ViT 每层，信号持续 |

**与传统方案的区别**：
- 区别于 QA-ViT（冻结 ViT + 仅 2 层注入 + additive bias）：本方案全层 KV injection，信号直接参与 attention
- 区别于 ViLT（从头训练）：从预训练 ViT 初始化，gate=0 保证安全启动

### 方案 B：给 LLM 注入 2D 位置信息（Spatial 2D RoPE）

**核心思路**：给 LLM 中的视觉 token 使用 2D 旋转位置编码，替代默认的 1D 序列位置编码，让 LLM 的 self-attention 天然感知 patch 的 2D 空间邻接关系。

- 视觉 token：将 head_dim 拆成四份，前半编码 row 位置，后半编码 col 位置
- 文本 token：保持标准 1D RoPE 不变
- 通过 `patch_llm_with_spatial_rope()` monkey-patch 到 LLM 的 rotary embedding 中
- LLM 在做 self-attention 时，天然知道哪些 patch 是空间相邻的

### 方案 C：Latent Reasoning Loop（隐空间迭代推理）

**核心思路**：模拟人类"逐个看、然后串联"的空间推理过程。人类看"苹果和香蕉哪个离杯子远"时，会依次聚焦苹果、香蕉、杯子的位置，存入工作记忆，最后比较。本方案在 ViT 的隐空间中实现这个迭代过程。

- ViT 只跑一次，产出 patch features（frozen after encoding）
- 一组**可学习的 latent tokens**（K=8）在隐空间中迭代推理（N=4 轮）：
  - **Cross-Attn to Text** → 自主学习"接下来该关注什么"（不需要显式实体解析）
  - **Cross-Attn to Patches** → 从图像对应区域读取视觉信息
  - **Self-Attn** → 整合多轮积累的空间信息
- Latent tokens 有状态，每轮携带前轮记忆，逐步积累空间理解
- **不依赖具体实体名称**，学到的是"怎么根据文本分步地看图"这个通用能力
- 所有迭代共享权重（类 RNN 展开），用 iteration embedding 区分第几轮

**与方案 C 传统版（Spatial-Aware Cross Attention）的区别**：
- 传统版：单次 cross-attention + spatial bias → 浅层空间匹配
- Latent Reasoning Loop：多轮迭代 + 记忆积累 → 深层空间推理

**LLM 输入序列**：`[visual_tokens (729) | reasoning_tokens (8) | text_tokens (T)]`

## 项目结构

```
vlm/
├── configs/
│   └── default.yaml                          # 模型配置、训练超参数、数据集路径
├── src/
│   ├── model/
│   │   ├── text_conditioned_vit/
│   │   │   ├── text_conditioned_vit.py       # 方案 A 传统实现：TextInjectionLayer + TextConditionedViT
│   │   │   └── text_conditioned_vit_kv.py    # 方案 A KV Injection：TextContextEncoder + KVInjectionHead + TextConditionedViTKV
│   │   ├── spatial_rope/
│   │   │   └── spatial_2d_rope.py            # 方案 B：Spatial2DRoPE + HybridPositionEmbedding
│   │   ├── spatial_cross_attention/
│   │   │   └── spatial_cross_attention.py    # 方案 C 传统版（已被 Latent Reasoning Loop 替代）
│   │   ├── latent_reasoning/
│   │   │   └── latent_reasoning_loop.py      # 方案 C：ReasoningBlock + LatentReasoningLoop
│   │   └── spatial_vlm/
│   │       └── spatial_vlm.py                # 整合模型：SpatialVLM
│   ├── data/                                 # 数据加载与预处理
│   │   └── pretrain_dataset.py               # 预训练数据集：SpatialPretrainDataset
│   ├── training/                             # 训练流程
│   │   └── pretrain_spatial.py               # 预训练：SpatialMaskGenerator + LatentPredictor + TextConditionedPretrainer
│   └── evaluation/                           # 评估脚本
└── requirements.txt
```

## 模型 Pipeline

```
Input Image + Text Query
        │
        ├──────────────────────┐
        ▼                      ▼
┌──────────────┐    ┌──────────────────┐
│ Text Encoder │    │ 方案 A:           │
│  (frozen)    │───▶│ Text-Conditioned │ ── 文本 compress 为 context (256d)
│              │    │ ViT (KV Inject)  │    注入每层 attention 的 KV cache
└──────┬───────┘    └────────┬─────────┘
       │                     │ visual_features
       │                     ▼
       │            ┌──────────────────┐
       │            │ Visual Projector │ ── 投影到 LLM 空间
       │            └────────┬─────────┘
       │                     │
       ▼                     ▼
┌─────────────────────────────────────┐
│ 方案 C: Latent Reasoning Loop       │ ── latent tokens 迭代推理
│ cross-attn(text) → cross-attn(img) │    4 轮迭代，逐步积累
│ → self-attn → FFN (×4 iterations)  │    空间关系信息
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ LLM Decoder                         │
│ 方案 B: 视觉 token → 2D RoPE        │ ── LLM 天然感知 2D 空间位置
│         文本 token → 1D RoPE        │
└──────────────────┬──────────────────┘
                   │
                   ▼
              Output Text
```

## 训练策略

### 阶段 0：Text-Conditioned Latent Prediction 预训练（KV Injection 模块）

**核心创新**：将 V-JEPA 的隐空间预测与文本条件化 KV Injection 结合——V-JEPA 证明隐空间预测能学到空间结构，但它是无条件的，接 VLM 后提升有限。我们加上文本条件化，让预训练直接学到"带着问题去看"的能力。

```
Target Encoder (EMA):  原始 ViT → 完整图像 → target features
Context Encoder:       TextConditionedViTKV → 部分可见 + 文本 KV → context features
Predictor (3层):       context + mask tokens + 文本 cross-attn → 预测遮挡区域的 latent features
Loss:                  Smooth L1 on L2-normalized features（只在遮挡位置计算）
```

**文本条件化空间掩码**（区别于 V-JEPA 的随机掩码）：
- 文本含"左边" → 遮左半区域 → 模型必须用文本空间线索预测
- 文本含 bbox → 遮 bbox 覆盖区域（block masking）
- 25% 随机掩码作为正则化
- 三种策略按概率混合：directional 50% / block 25% / random 25%

**数据**：RefCOCO/RefCOCOg（referring expression + bbox，天然空间描述）

**预训练后**：只保留 TextContextEncoder + KVInjectionHeads 迁移到 SpatialVLM。Predictor 和 target encoder 丢弃。

### 阶段 1：对齐预训练
- **冻结**：ViT + LLM
- **训练**：TextContextEncoder、KVInjectionHead（×27 层）、Visual Projector、SpatialAwareCrossAttention、text_projector
- **初始化**：从阶段 0 加载预训练的 KV Injection 权重
- 学习率：1e-3，batch size：256

### 阶段 2：全量微调
- **解冻**：所有模块（包括 ViT，让文本信号反向传播调整视觉编码）
- 学习率：2e-5，batch size：64

## 评估 Benchmark

- **VSI-Bench**：视觉空间推理
- **SpatialQA**：空间问答
- **Theory-of-Space**：空间感知综合评估（Active-Passive Gap、Belief Drift、Belief Inertia）
- **PerceptionTest**：通用视觉感知
- **TempCompass**：时序理解

## 安装

```bash
pip install -r requirements.txt
```

## 相关论文

- **DFlash (2026)**：block diffusion 加速推理，通过 KV injection 将 target model context feature 注入 drafter 每层，实现持续条件化。**方案 A KV injection 的核心灵感来源**
- **ViLT (ICML 2021)**：image patches 和 text tokens 联合 self-attention，方案 A 的早期参考
- **QA-ViT (CVPR 2024)**：在 ViT 层间注入文本 token（冻结 ViT），方案 A 在此基础上改进为全层 KV injection
- **Theory of Space (ICLR 2026)**：VLM 空间感知能力评估框架，发现三大核心缺陷
- **V-JEPA 2 (Meta, 2025)**：自监督视频模型，证明隐空间预测能学到 3D 空间结构，但无条件化导致接 LLM 后提升有限。**阶段 0 预训练的核心灵感来源**——我们加上文本条件化解决这一局限
- **Spatial-SSRL (2025)**：空间自监督预训练任务，效果有限（+4.63%），佐证训练信号路线的局限性
