# High-Bandwidth Data Plan

## 1. Core Position

The real bottleneck is not only image count or visual token count.

The bottleneck is **per-image supervision bandwidth**.

A modern VLM is often trained with:

```text
1 image -> 1 short caption or 1 short QA pair
```

This underutilizes the image. A single image can support many more supervision units:

- object references
- attributes
- relations
- comparisons
- counts
- negative cases
- reasoning traces

The goal of this data plan is therefore:

```text
1 scene -> K questions -> multiple structured supervision targets
```

## 2. Four Supervision Layers

Each scene should ideally support supervision at four levels.

### A. Entity Layer

- object identity
- color / shape / size
- box / point / mask-like localization
- existence / count

### B. Relation Layer

- left / right
- above / below
- overlap / inside
- closer / farther
- same size / different size

### C. Composition Layer

- conjunctions
- chained references
- attribute-constrained counting
- multi-hop comparisons

### D. Trace Layer

- draft claim
- decomposed subclaims
- oracle reasoning trace
- refusal or zero-answer traces

## 3. Synthetic-First Strategy

We start with synthetic scenes because they provide:

- clean object identities
- exact object boxes
- programmatically verified relations
- scalable negative construction
- reproducible trace generation

The synthetic generator should not produce just one QA pair per image. It should produce:

```text
1 image -> 4 or more structured questions
```

This is the minimum practical step toward higher supervision bandwidth.

## 4. Anti-Pattern Learning Rules

Synthetic data is useful only if it avoids shallow patterns.

The generator should randomize:

### Scene randomness

- object count
- object density
- object size distribution
- overlap level
- margin to image boundary

### Appearance randomness

- colors
- shapes
- backgrounds
- style noise
- optional blur / compression in later versions

### Structural randomness

- sparse vs dense layouts
- symmetric vs asymmetric arrangements
- near-boundary relations
- ambiguous distractors

### Language randomness

- multiple question templates per task
- yes/no vs wh questions
- positive vs negative phrasings

## 5. Synthetic Curriculum

### Phase A

Atomic tasks:

- `left_of`
- `right_of`
- `above`
- `below`
- `count`

### Phase B

Compositional tasks:

- conjunction
- chained reference
- compare distance

### Phase C

Hard negatives:

- no matching object
- zero-count cases
- misleading nearby distractors

## 6. Real-Data Follow-Up

After synthetic warm start, real data should be added in increasing structure order:

1. bbox-grounded data
2. scene-graph-backed data
3. dense counting / chart / GUI / document data
4. open-ended instruction data

The real-data stage should preserve the same principle:

**do not let one image collapse back into one weak text target.**

## 7. Interaction with QCR

The data line and the QCR line should run in parallel.

This gives a clean experiment grid:

1. baseline + weak data
2. baseline + high-bandwidth data
3. QCR + weak data
4. QCR + high-bandwidth data

This matrix can answer:

- whether data density alone helps
- whether QCR alone helps
- whether the two are complementary

## 8. Immediate Batch Plan

The small-batch synthetic config uses:

- 48 train scenes
- 12 val scenes
- 12 test scenes
- 4 questions per scene

This yields:

- 192 train samples
- 48 val samples
- 48 test samples

This is enough for a first sanity-check batch and schema inspection.
