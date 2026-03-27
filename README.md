# SpatialVLM: Spatial-Aware Vision-Language Model with 3D Belief Memory

> **Towards Deep Spatial Intelligence in Vision-Language Models**
>
> 当前主流 VLM（ViT + Projector + LLM Decoder）在空间感知上存在根本性架构瓶颈。
> 本项目旨在通过 **3D-aware 视觉编码**、**双坐标系位置编码** 和 **空间信念记忆机制**，
> 赋予 VLM 真正的空间理解能力。

## Motivation

### 问题：当前 VLM 为什么做不好空间感知？

根据 [Theory of Space (ICLR 2026)](https://arxiv.org/abs/2602.07055) 的系统性评估，当前 SOTA VLM 在空间任务上暴露出三大核心缺陷：

| 缺陷 | 表现 | 根因 |
|------|------|------|
| **Perception Bottleneck** | Vision 世界准确率仅 32-52%，远低于 Text 世界的 80% | ViT 的 2D patch 编码丢失了 3D 空间信息 |
| **Belief Drift** | 正确感知的空间信息在后续步骤中被错误覆盖 | 缺乏稳定的空间记忆机制，信息在 token 序列中被冲刷 |
| **Belief Inertia** | 环境变化后无法更新已有的空间信念（朝向惯性高达 68.9%） | 缺乏动态信念修正机制 |

### 核心观点

> 问题不仅仅是"看不准"（Perception），更严重的是"记不住"（Drift）和"改不了"（Inertia）。
> 根因在于——**缺乏显式的、可更新的空间世界模型**。

---

## Architecture

### 整体架构

```
Input Image
    │
    ▼
┌──────────────────────────┐
│  3D-aware ViT            │  ← Module 1: NeRF-like 隐式 3D 表征
│  ├── 2D Patch Embedding  │
│  ├── Depth Branch        │     并行深度估计分支
│  └── 3D Feature Fusion   │     融合 RGB + Depth → 3D-aware tokens
└────────────┬─────────────┘
             │ 3D-aware visual tokens
             ▼
┌──────────────────────────┐
│  Dual Coordinate         │  ← Module 2: 双坐标系位置编码
│  Position Encoding       │
│  ├── Ego-centric PE      │     自我中心坐标（导航/路径推理）
│  ├── Allocentric PE      │     世界中心坐标（全局地图构建）
│  └── Learnable Switch    │     可学习的坐标系切换门控
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐     ┌─────────────────────────────┐
│  LLM Decoder             │◄───►│  Spatial Belief Memory      │ ← Module 3
│  (Frozen / Fine-tuned)   │     │  ├── Scene Graph Store      │
│                          │     │  │   Nodes: obj → (pos, ori, │
│  每 k 层插入:             │     │  │     conf, appearance, ts) │
│  ├── Cross-Attn to       │     │  │   Edges: (i,j) → relation │
│  │   Spatial Memory      │     │  ├── Confidence-Gated Write │
│  └── Uncertainty Signal  │     │  └── Uncertainty-Aware Read │
└────────────┬─────────────┘     └─────────────────────────────┘
             │
             ▼
    Output (text / action / cognitive map)
```

### Module 1: 3D-aware ViT

**目标**：从单张 2D 图像中提取 3D 空间感知的视觉 token。

**设计**：
- 在标准 ViT 的基础上，并行一个 **Depth Estimation Branch**（基于 Depth Anything V2 初始化）
- 每个 patch 不仅有 RGB 特征，还关联一个估计的深度值 z
- 通过 **3D Feature Fusion Module** 将 (RGB_feature, depth_feature) 融合为 3D-aware token
- 融合方式：Gated Cross-Attention，让 RGB 和 Depth 特征互相增强

```python
class ThreeDawareViT(nn.Module):
    def __init__(self, base_vit, depth_encoder):
        self.rgb_encoder = base_vit           # 预训练 ViT (e.g., SigLIP)
        self.depth_encoder = depth_encoder     # 预训练 Depth Anything V2
        self.fusion = GatedCrossAttention(dim=hidden_dim)

    def forward(self, image):
        rgb_tokens = self.rgb_encoder(image)        # [B, N, D]
        depth_map = self.depth_encoder(image)        # [B, H, W, 1]
        depth_tokens = patchify_depth(depth_map)     # [B, N, D_depth]
        fused_tokens = self.fusion(rgb_tokens, depth_tokens)  # [B, N, D]
        return fused_tokens, depth_map
```

**关键点**：
- Depth Branch 用 Depth Anything V2 初始化，提供强先验
- 融合模块是可学习的，让模型自己决定 RGB 和 Depth 信息的权重
- 输出的每个 token 都隐式包含了 3D 位置信息

### Module 2: Dual Coordinate Position Encoding

**目标**：让模型同时维护自我中心和世界中心两种空间表征。

**设计**：

```python
class DualCoordinatePE(nn.Module):
    def __init__(self, dim, max_positions=1024):
        # Ego-centric: 以当前视角为原点的 3D RoPE
        self.ego_rope = Rotary3DPositionEncoding(dim)
        # Allocentric: 以世界坐标系为原点的 3D RoPE
        self.allo_rope = Rotary3DPositionEncoding(dim)
        # 可学习的坐标系切换门控
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, tokens, ego_coords_3d, allo_coords_3d):
        ego_pe = self.ego_rope(ego_coords_3d)    # 自我中心位置编码
        allo_pe = self.allo_rope(allo_coords_3d)  # 世界中心位置编码
        gate = self.gate(tokens)
        return tokens + gate * ego_pe + (1 - gate) * allo_pe
```

**Ego-centric 坐标**：
- 以当前 agent 位置为原点
- (x, y) 来自 patch 在图像中的位置，z 来自 depth 估计
- 适用于 Route-level 任务（路径推理、导航）

**Allocentric 坐标**：
- 以世界坐标系为原点
- 需要结合 agent 的历史轨迹和 IMU/pose 信息推算
- 适用于 Survey-level 任务（全局地图构建）

**Learnable Switch**：
- 不同的 attention layer 可能需要不同坐标系的信息
- 门控机制让模型自动学习在什么情况下用哪种坐标系

### Module 3: Spatial Belief Memory

**目标**：解决 Belief Drift 和 Belief Inertia，提供稳定、可更新的空间记忆。

#### 3.1 Scene Graph Memory Store

```python
class SpatialSceneGraph:
    """
    显式的空间场景图记忆。
    每个节点代表一个物体，每条边代表两个物体之间的空间关系。
    """
    nodes: Dict[str, ObjectNode]
    # ObjectNode = {
    #     position: Tensor[3],        # 3D 坐标 (x, y, z)
    #     orientation: Tensor[4],     # 四元数表示朝向
    #     appearance: Tensor[D],      # 外观特征向量
    #     confidence: float,          # 置信度 [0, 1]
    #     timestamp: int,             # 最后更新时间步
    #     observation_count: int      # 被观察次数
    # }

    edges: Dict[Tuple[str, str], RelationEdge]
    # RelationEdge = {
    #     relative_direction: Tensor[3],  # 相对方向向量
    #     relative_distance: float,       # 相对距离
    #     confidence: float
    # }

    agent_state: AgentState
    # AgentState = {
    #     position: Tensor[3],
    #     orientation: Tensor[4],
    #     timestamp: int
    # }
```

#### 3.2 Confidence-Gated Write

```python
class ConfidenceGatedWriter(nn.Module):
    """
    置信度门控写入机制。
    根据新旧信息的冲突程度和旧信息的置信度，决定如何更新记忆。
    """
    def __init__(self, dim):
        self.conflict_detector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.revision_gate = nn.Sequential(
            nn.Linear(dim * 2 + 2, dim),  # +2 for confidence and conflict
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def write(self, memory, new_observation, object_id):
        old_node = memory.nodes[object_id]

        # 1. 计算冲突程度
        conflict = self.conflict_detector(
            torch.cat([old_node.embedding, new_observation.embedding], dim=-1)
        )

        # 2. 门控决策
        revision_input = torch.cat([
            old_node.embedding,
            new_observation.embedding,
            old_node.confidence,
            conflict
        ], dim=-1)
        revision_weight = self.revision_gate(revision_input)

        # 3. 更新策略
        # revision_weight 高 → 用新观察替换旧信息（Belief Revision）
        # revision_weight 低 → 保持旧信息，微调（Belief Preservation）
        updated_embedding = (
            revision_weight * new_observation.embedding +
            (1 - revision_weight) * old_node.embedding
        )

        # 4. 更新置信度
        # 多次一致观察 → 置信度上升
        # 冲突观察 → 置信度先下降再恢复
        updated_confidence = self._update_confidence(
            old_node.confidence, conflict, old_node.observation_count
        )

        return updated_embedding, updated_confidence
```

#### 3.3 Uncertainty-Aware Read

```python
class UncertaintyAwareReader(nn.Module):
    """
    不确定性感知的记忆读取。
    在 LLM decoder 中通过 cross-attention 查询空间记忆，
    同时输出 uncertainty signal 指导主动探索。
    """
    def __init__(self, dim, num_heads=8):
        self.cross_attn = nn.MultiheadAttention(dim, num_heads)
        self.uncertainty_head = nn.Linear(dim, 1)

    def read(self, decoder_hidden, memory):
        # 将场景图节点转为 key-value
        memory_keys = memory.get_all_node_embeddings()      # [M, D]
        memory_values = memory.get_all_node_embeddings()     # [M, D]
        confidence_weights = memory.get_all_confidences()    # [M, 1]

        # Cross-attention with confidence weighting
        attn_output, attn_weights = self.cross_attn(
            query=decoder_hidden,
            key=memory_keys,
            value=memory_values
        )
        # 用置信度加权 attention 输出
        weighted_output = attn_output * confidence_weights.unsqueeze(0)

        # 计算 uncertainty signal
        uncertainty = self.uncertainty_head(weighted_output)  # [B, N, 1]

        return weighted_output, uncertainty
```

---

## Training Strategy

### 基座模型选择

| 候选模型 | 优势 | 劣势 |
|---------|------|------|
| **Qwen2.5-VL-7B** | 开源、中文友好、ViT 架构清晰 | 社区相对较小 |
| **InternVL2.5-8B** | 性能强、架构灵活 | 模型较大 |
| **LLaVA-NeXT-7B** | 社区活跃、代码清晰 | 空间能力一般 |

**推荐**：Qwen2.5-VL-7B 或 InternVL2.5-8B 作为起点。

### 三阶段增量训练

```
Stage 1: 3D Visual Grounding (3D 视觉对齐)
├── 冻结: LLM Decoder
├── 训练: 3D-aware ViT (Depth Branch + Fusion Module)
├── 数据: Objaverse 多视角渲染 + ScanNet 室内场景 + Depth Anything 伪标签
├── Loss: Depth Estimation Loss + 3D Feature Contrastive Loss
└── 目标: 让视觉编码器能从单张图提取 3D 空间信息

Stage 2: Spatial Memory Training (空间记忆训练)
├── 冻结: 3D-aware ViT
├── 训练: Spatial Belief Memory + Dual Coordinate PE + Projector
├── 数据: 构造序列化空间探索数据（多步观察 → 构建场景图）
├── Loss: Memory Reconstruction Loss + Belief Revision Loss
│         + Uncertainty Calibration Loss
└── 目标: 让记忆模块学会稳定存储和动态更新空间信息

Stage 3: End-to-End Fine-tuning (端到端微调)
├── 训练: 全参数（ViT + Memory + LLM，LLM 用 LoRA）
├── 数据: Theory of Space benchmark 数据
│         + 空间推理 instruction tuning 数据
│         + 主动探索 + 信念修正数据
├── Loss: Task Loss + Exploration Reward (信息增益)
└── 目标: 端到端优化整个系统的空间智能
```

### 数据准备

#### Stage 1 数据
- **Objaverse 多视角渲染**：从 Objaverse 3D 资产渲染多视角 RGB + Depth 图像对
- **ScanNet / ScanNet++**：真实室内场景的 RGB-D 数据
- **Depth Anything V2 伪标签**：对大规模图像数据生成深度伪标签

#### Stage 2 数据
- **序列化空间探索数据**：模拟 agent 在室内场景中移动，记录每步的观察和场景图变化
- **信念修正数据**：构造"环境变化"场景，要求模型更新空间记忆
- 可使用 **ThreeDWorld** 或 **Habitat** 模拟器生成

#### Stage 3 数据
- **Theory of Space benchmark**：论文提供的评估数据（[HuggingFace](https://huggingface.co/datasets/MLL-Lab/tos-data)）
- **空间推理 instruction tuning**：构造空间推理 QA 对
- **主动探索数据**：agent 自主探索 + 信息增益奖励

---

## Evaluation

### 主要评估基准

1. **Theory of Space Benchmark** (ICLR 2026)
   - Route-level 任务：Pairwise Direction, Perspective Taking, Action2View, View2Action
   - Survey-level 任务：Allocentric Map, Mental Rotation, Location2View, View2Location
   - 探索效率：步数、信息增益曲线
   - 信念质量：Cognitive Map Probing（Correctness, Stability, Perception, Self-tracking）
   - 信念修正：False Belief 范式（Belief Inertia 指标）

2. **补充评估**
   - ScanQA：3D 场景问答
   - SQA3D：Situated Question Answering in 3D
   - VSI-Bench：Video Spatial Intelligence

### 目标指标

| 指标 | 当前 SOTA (Gemini-3 Pro) | 我们的目标 |
|------|------------------------|-----------|
| Active Exploration Avg | 57.3% | **70%+** |
| Cognitive Map Correctness (Vision) | 52.1% | **75%+** |
| Belief Stability (Vision) | ~62% | **85%+** |
| Belief Inertia (Vision, ↓) | 51.1% (ori) | **<20%** |
| Active-Passive Gap | ~3-11% | **<3%** |

---

## Project Structure

```
vlm/
├── README.md
├── configs/                          # 训练配置
│   ├── stage1_3d_grounding.yaml
│   ├── stage2_spatial_memory.yaml
│   └── stage3_end2end.yaml
├── src/
│   ├── model/
│   │   ├── three_d_aware_vit.py      # Module 1: 3D-aware ViT
│   │   ├── depth_branch.py           # Depth Estimation Branch
│   │   ├── dual_coordinate_pe.py     # Module 2: 双坐标系位置编码
│   │   ├── spatial_belief_memory.py  # Module 3: 空间信念记忆
│   │   │   ├── scene_graph.py        #   场景图存储
│   │   │   ├── confidence_writer.py  #   置信度门控写入
│   │   │   └── uncertainty_reader.py #   不确定性感知读取
│   │   ├── spatial_vlm.py            # 完整模型组装
│   │   └── projector.py              # 视觉-语言投影层
│   ├── data/
│   │   ├── objaverse_dataset.py      # Stage 1 数据
│   │   ├── spatial_exploration.py    # Stage 2 数据
│   │   └── tos_benchmark.py          # Stage 3 / 评估数据
│   ├── training/
│   │   ├── stage1_trainer.py
│   │   ├── stage2_trainer.py
│   │   └── stage3_trainer.py
│   └── evaluation/
│       ├── tos_evaluator.py          # Theory of Space 评估
│       ├── belief_probing.py         # 认知地图探测
│       └── metrics.py                # 评估指标
├── scripts/
│   ├── train_stage1.sh
│   ├── train_stage2.sh
│   ├── train_stage3.sh
│   └── evaluate.sh
├── requirements.txt
└── 2602.07055v1.pdf                  # Theory of Space 论文
```

---

## Quick Start

### 环境安装

```bash
pip install -r requirements.txt
```

### 训练

```bash
# Stage 1: 3D Visual Grounding
bash scripts/train_stage1.sh

# Stage 2: Spatial Memory Training
bash scripts/train_stage2.sh

# Stage 3: End-to-End Fine-tuning
bash scripts/train_stage3.sh
```

### 评估

```bash
# Theory of Space Benchmark
bash scripts/evaluate.sh
```

---

## Key References

- **Theory of Space** (ICLR 2026): [arXiv:2602.07055](https://arxiv.org/abs/2602.07055) | [GitHub](https://github.com/mll-lab-nu/Theory-of-Space) | [Data](https://huggingface.co/datasets/MLL-Lab/tos-data)
- **Depth Anything V2**: [arXiv:2406.09414](https://arxiv.org/abs/2406.09414)
- **Qwen2.5-VL**: [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
- **NeRF**: [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)
- **SpatialRGPT**: [arXiv:2406.01584](https://arxiv.org/abs/2406.01584)
- **SpatialVLM (Chen et al.)**: [arXiv:2401.12168](https://arxiv.org/abs/2401.12168)

---

## License

Apache 2.0
