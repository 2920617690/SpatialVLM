# Synthetic Data Schema

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

Each sample stores:

```text
sample_id
scene_id
split
image_path
task_type
question
answer
draft_response
verify_response
final_response
subclaims
scene_objects
metadata
```

Each rendered scene now supports multiple question instances.

The default generator uses:

```text
1 scene -> 4 questions
```

so the actual number of samples is:

```text
num_samples = num_scenes * questions_per_scene
```

The current QCR code uses:

- `question`
- `draft_response`
- `final_response`
- `image_path`

The extra fields remain because they are still useful for later inspect-style extensions and analysis.
