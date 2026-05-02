# Blender CLEVR-Style Pipeline

This is the preferred Blender path for richer synthetic supervision.

## Why This Version

Compared with arbitrary `.glb` assets, procedural geometric scenes are:

- easier to randomize
- easier to annotate
- cleaner for relation reasoning
- closer to the visual style used in CLEVR-like reasoning work

## Scene Contents

Each scene contains:

- 4 to 7 objects
- shape from `{cube, sphere, cylinder, cone}`
- material from `{rubber, metal}`
- color from a fixed palette
- size from `{small, medium, large}`

## Exported Metadata

Per object:

- object id
- descriptor
- color
- material
- shape
- size
- 2D normalized bbox
- 2D center
- 3D location
- scale

Per scene:

- image path
- camera pose
- split and scene id

## Output Layout

```text
data/synthetic/qwen35_qcr_blender_clevr_batch/
├── images/
├── scene_metadata/
├── manifests/
└── metadata/
```

## Command

```bash
python3 scripts/synthesize_blender_clevr.py --config configs/blender_clevr_batch.json
```
