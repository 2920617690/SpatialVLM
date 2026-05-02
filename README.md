# QCR for Qwen3.5-4B

This repository is rewritten around a smaller and cleaner idea:

- base model: `Qwen3.5-4B`
- backbone assumption: `ViT + projector + decoder-only LLM`
- method: `Query-Conditioned Re-encoding (QCR)`

The central question is:

**does it help if the same image is encoded a second time after the model already knows the question?**

## Repository Layout

```text
vlm/
├── configs/
│   ├── base.yaml
│   ├── baseline_sft.yaml
│   ├── qcr_sft.yaml
│   └── synth_data.yaml
├── docs/
│   ├── data_schema.md
│   ├── method.md
│   └── method_zh.md
├── scripts/
│   ├── build_relation_data.py
│   ├── synthesize_blender_clevr.py
│   ├── synthesize_blender_light.py
│   ├── synthesize_data.py
│   ├── train_baseline.py
│   └── train_sft.py
├── src/
│   ├── data/
│   ├── model/
│   └── train/
└── requirements.txt
```

## Two Core Experiments

The repo is now organized around two models:

1. `one-pass baseline`
2. `QCR two-pass`

### One-pass baseline

```text
image -> ViT -> projector -> decoder -> answer
```

### QCR two-pass

```text
pass 1:
  image -> ViT -> projector -> decoder
  read hidden state at <reencode_slot>

pass 2:
  project that hidden state into a condition token
  prepend it to the same image token sequence
  run a second encoding pass
  residual-refine first-pass visual tokens
  decode final answer
```

## Data Path

The default synthetic root is:

```text
data/synthetic/qwen35_qcr/
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

See [docs/data_schema.md](/Users/fwk/Downloads/vlm/docs/data_schema.md) for the manifest schema.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Synthesize data

```bash
python3 scripts/synthesize_data.py --config configs/synth_data.yaml
```

### 2b. Synthesize Blender-light data

```bash
python3 scripts/synthesize_blender_light.py --config configs/blender_light_batch.json
```

### 2c. Synthesize Blender CLEVR-style data

```bash
python3 scripts/synthesize_blender_clevr.py --config configs/blender_clevr_batch.json
```

### 3. Train the one-pass baseline

```bash
python3 scripts/train_baseline.py --config configs/baseline_sft.yaml
```

### 4. Train QCR

```bash
python3 scripts/train_sft.py --config configs/qcr_sft.yaml
```

## Docs

- English method note: [docs/method.md](/Users/fwk/Downloads/vlm/docs/method.md)
- 中文方法说明: [docs/method_zh.md](/Users/fwk/Downloads/vlm/docs/method_zh.md)
- Data schema: [docs/data_schema.md](/Users/fwk/Downloads/vlm/docs/data_schema.md)
- High-bandwidth data plan: [docs/high_bandwidth_data.md](/Users/fwk/Downloads/vlm/docs/high_bandwidth_data.md)
- Blender-light pipeline: [docs/blender_light_pipeline.md](/Users/fwk/Downloads/vlm/docs/blender_light_pipeline.md)
- Blender CLEVR-style pipeline: [docs/blender_clevr_pipeline.md](/Users/fwk/Downloads/vlm/docs/blender_clevr_pipeline.md)

## Practical Notes

- The config defaults to `Qwen/Qwen3.5-4B`.
- `train_baseline.py` is the cleanest standard path.
- `train_sft.py` is the QCR path.
- The code tries to run strict same-image re-encoding through a shared vision tower interface. If the loaded checkpoint does not expose the required internals, it falls back to projected-token re-encoding and warns explicitly.
