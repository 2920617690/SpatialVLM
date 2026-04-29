# Data Synthesis Path

The default synthetic data root is:

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

## Scene Content

Each image contains 3 to 6 synthetic objects with:

- shape
- color
- size
- bounding box
- center

## Task Mix

The current generator supports:

- `atomic_yesno`
- `conjunction_yesno`
- `reference_yesno`
- `compare_distance_yesno`
- `count_relation`
- `chain_relation_yesno`

## Manifest Schema

Each jsonl row contains:

```text
sample_id
split
image_path
task_type
question
answer
draft_response
verify_response
final_response
subclaims
trajectory
scene_objects
metadata
```

## Generation Command

```bash
python3 scripts/synthesize_data.py --config configs/synth_data.yaml
```

## Real Data Path

The repo also keeps a relation builder for bbox-based datasets:

```bash
python3 scripts/build_relation_data.py --instances path/to/instances.json --output data/real/train_relations.jsonl
```

That path is mainly useful for later transfer, not for the initial self-verification experiments.
