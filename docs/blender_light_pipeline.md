# Blender-Light Pipeline

This is an Infinigen-inspired but much lighter pipeline for generating structured supervision data with Blender.

## Goal

Generate small but visually richer scenes than the current 2D geometry generator while preserving:

- object identity
- 2D boxes
- object attributes
- scene-level metadata
- multi-question supervision per rendered image

## Pipeline

### Stage 1: Blender render

Blender renders a small scene and exports:

- RGB image
- per-scene metadata json

Each metadata json includes:

- object ids
- asset keys
- categories
- colors
- 2D normalized boxes
- 2D centers
- 3D positions
- 3D scales

### Stage 2: Sample generation

The host Python script converts each scene metadata file into multiple structured supervision samples using the same task family as the current 2D generator:

- atomic relation
- conjunction
- reference
- compare distance
- count
- chain relation

## Output Layout

```text
data/synthetic/qwen35_qcr_blender_batch/
├── images/
├── scene_metadata/
├── manifests/
└── metadata/
```

## First Version Design Constraints

- small asset pool
- small object count
- strong randomization over object color, placement, light, and camera
- multi-question supervision per scene

This is not meant to replace Infinigen. It is meant to provide a controlled intermediate step between very simple 2D scenes and full procedural worlds.
