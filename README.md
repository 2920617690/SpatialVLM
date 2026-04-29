# AVV for Qwen3.5-4B

This repository targets a concrete experimental setup:

- base model: `Qwen3.5-4B`
- backbone assumption: `ViT + projector + decoder-only LLM`
- training objective: teach the same VLM to execute `draft -> verify -> final`
- policy objective: learn `when to verify`, `when to revise`, and `when to stop`

The current codebase is organized to support:

1. synthetic spatial data generation
2. stage-0 supervised warm start
3. stage-1 oracle-guided imitation learning
4. stage-2 budgeted verification policy optimization (BVPO)

## Repository Layout

```text
vlm/
├── configs/
│   ├── base.yaml
│   ├── stage0_sft.yaml
│   ├── stage1_imitation.yaml
│   ├── stage2_bvpo.yaml
│   └── synth_data.yaml
├── docs/
│   ├── data_pipeline.md
│   ├── method.md
│   └── method_zh.md
├── scripts/
│   ├── build_relation_data.py
│   ├── synthesize_data.py
│   ├── train_imitation.py
│   ├── train_policy.py
│   └── train_sft.py
├── src/
│   ├── data/
│   ├── model/
│   ├── rl/
│   └── train/
└── requirements.txt
```

## Main Idea

Instead of attaching heavy extra modules to the VLM, this repo uses the same multimodal model across multiple modes:

- `draft mode`: produce a structured draft answer and draft claim
- `verify mode`: judge whether the claim is supported by the image
- `final mode`: produce the final answer after seeing the verification result
- `policy mode`: decide the next high-level action among `PROPOSE`, `VERIFY`, `REVISE`, `ANSWER`, `ABSTAIN`

## Data Path

The default synthetic data path is:

```text
data/synthetic/qwen35_avv/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── manifests/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── metadata/
    ├── train_summary.json
    ├── val_summary.json
    └── test_summary.json
```

See [docs/data_pipeline.md](/Users/fwk/Downloads/vlm/docs/data_pipeline.md) for the full synthesis path and schema.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Synthesize data

```bash
python3 scripts/synthesize_data.py --config configs/synth_data.yaml
```

### 3. Stage-0 SFT

```bash
python3 scripts/train_sft.py --config configs/stage0_sft.yaml
```

### 4. Stage-1 Imitation

```bash
python3 scripts/train_imitation.py --config configs/stage1_imitation.yaml
```

### 5. Stage-2 BVPO

```bash
python3 scripts/train_policy.py --config configs/stage2_bvpo.yaml
```

## Docs

- English method note: [docs/method.md](/Users/fwk/Downloads/vlm/docs/method.md)
- 中文方法说明: [docs/method_zh.md](/Users/fwk/Downloads/vlm/docs/method_zh.md)
- Data synthesis path and schema: [docs/data_pipeline.md](/Users/fwk/Downloads/vlm/docs/data_pipeline.md)

## Practical Notes

- The config defaults to `Qwen/Qwen3.5-4B`. If your actual checkpoint id differs, change `model.base_model_id`.
- The stage-0 and stage-1 code paths are the most concrete parts of the repo.
- The stage-2 BVPO implementation is intentionally lightweight and research-oriented. It is meant as an experimental starting point, not a production RL trainer.
