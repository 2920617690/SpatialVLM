# AVV: Answer, Visual Evidence, Verify

## 1. Problem Setup

Given an image `x` and a question `q`, standard VLMs directly model:

\[
p(y \mid x, q)
\]

where `y` is the answer text. This objective is underconstrained for spatial reasoning. It rewards the model for producing the correct answer, but does not require the model to preserve or check the visual evidence that justifies the answer.

We instead treat spatial QA as a **hypothesis verification** problem. The model first proposes an answer `y_0`, then gathers localized evidence and verifies whether that evidence supports the implied spatial claim before emitting a final answer `y^*`.

## 2. Core Hypothesis

Spatial failures often arise because the model commits too early:

- it binds the wrong instance,
- reads the wrong local region,
- compresses the wrong visual detail into language,
- and never visually checks its own claim.

The main intervention is therefore small but structural:

**do not answer directly; answer only after a visual verification step.**

## 3. Overview

AVV has five stages:

1. `Proposal`: produce an initial answer `y_0`.
2. `Evidence Query`: derive a compact query `z_0` describing what evidence would justify `y_0`.
3. `Evidence Localization`: predict evidence regions for object `A` and object `B`.
4. `Re-Perception`: crop those regions from the original image and re-encode them at higher fidelity.
5. `Verification`: decide whether the re-read evidence supports, contradicts, or is insufficient for the proposed claim.

The final decision head combines the proposal confidence and verification output to produce `y^*`.

## 4. Architecture

### 4.1 Backbone

The backbone encodes:

- image tokens `V = \{v_i\}_{i=1}^N`
- question representation `q_h`

The framework does not assume a specific VLM family. The only requirement is access to image features and a question-conditioned context vector.

### 4.2 Proposal Head

The proposal head produces:

\[
p_\theta(y_0 \mid x, q)
\]

and an evidence query vector:

\[
z_0 = f_{\text{evidence}}(V, q_h, y_0)
\]

`z_0` is not a full chain-of-thought. It is a compact latent descriptor of what the model believes should be checked next.

### 4.3 Evidence Pointer

The pointer head predicts evidence regions:

\[
b_A, b_B = f_{\text{ptr}}(V, q_h, z_0)
\]

where `b_A` and `b_B` are boxes or heatmaps corresponding to the two object regions that matter for the spatial claim.

### 4.4 Re-Perception Module

The key difference from latent-only reasoning is that AVV returns to pixels:

\[
\hat{x}_A = \text{crop}(x, b_A), \quad \hat{x}_B = \text{crop}(x, b_B)
\]

These crops are re-encoded to obtain evidence features:

\[
f_A = g(\hat{x}_A), \quad f_B = g(\hat{x}_B)
\]

This step is meant to correct first-pass errors caused by patch boundaries, coarse readout, or wrong instance binding.

### 4.5 Hypothesis Builder

The pair `(q, y_0)` is converted into a structured claim:

\[
c = h(q, y_0)
\]

Examples:

- `Is the cat left of the dog?` + `yes` -> `cat left of dog`
- `Is the apple farther than the banana from the cup?` + `no` -> `apple is not farther than banana from cup`

### 4.6 Relation Verifier

The verifier predicts:

\[
p_\theta(v \mid f_A, f_B, c)
\]

where:

\[
v \in \{\text{support}, \text{contradict}, \text{insufficient}\}
\]

The verifier is deliberately discriminative rather than generative. It should answer one narrow question: does the localized evidence support the proposed claim?

### 4.7 Final Decision Head

The final answer is computed from proposal confidence and verification state:

\[
p_\theta(y^* \mid y_0, v, q_h)
\]

In the simplest case:

- keep `y_0` if `v = support`,
- flip or revise if `v = contradict`,
- abstain or trigger fallback if `v = insufficient`.

## 5. Training Objective

The total objective is:

\[
\mathcal{L} =
\mathcal{L}_{ans} +
\lambda_{final}\mathcal{L}_{final} +
\lambda_{ptr}\mathcal{L}_{ptr} +
\lambda_{ver}\mathcal{L}_{ver} +
\lambda_{cf}\mathcal{L}_{cf} +
\lambda_{cons}\mathcal{L}_{cons}
\]

### 5.1 Proposal Loss

\[
\mathcal{L}_{ans} = \mathrm{CE}(p_\theta(y_0 \mid x, q), y)
\]

### 5.2 Final Answer Loss

\[
\mathcal{L}_{final} = \mathrm{CE}(p_\theta(y^* \mid x, q), y)
\]

### 5.3 Pointer Loss

For box supervision:

\[
\mathcal{L}_{ptr} =
\mathrm{L1}(b_A, b_A^*) + \mathrm{L1}(b_B, b_B^*) +
\mathrm{GIoU}(b_A, b_A^*) + \mathrm{GIoU}(b_B, b_B^*)
\]

For heatmap supervision, BCE can replace the box losses.

### 5.4 Verifier Loss

\[
\mathcal{L}_{ver} = \mathrm{CE}(p_\theta(v \mid f_A, f_B, c), v^*)
\]

### 5.5 Counterfactual Loss

For the same evidence pair, construct an incorrect relation claim `\tilde{c}` and require contradiction:

\[
\mathcal{L}_{cf} =
-\log p_\theta(v=\text{contradict} \mid f_A, f_B, \tilde{c})
\]

This discourages shortcut learning from object co-occurrence alone.

### 5.6 Consistency Loss

For a geometric transform `g` and the corresponding language transform `T_g`, require equivariant verification:

\[
\mathcal{L}_{cons} =
D\Big(
p_\theta(v \mid g(x), T_g(q)),
T_g\big(p_\theta(v \mid x, q)\big)
\Big)
\]

In practice, the first transform to use is horizontal flip with `left <-> right`.

## 6. Training Sample Construction

Each sample is promoted from `(image, question, answer)` to:

\[
(x, A, B, r, q, y, b_A^*, b_B^*)
\]

where:

- `A, B` are object instances,
- `r` is a geometric relation,
- `b_A^*, b_B^*` are evidence boxes.

### 6.1 Data Sources

Primary sources:

- COCO instances
- Visual Genome
- RefCOCO / RefCOCOg
- CLEVR or synthetic relation data

### 6.2 Relation Labels

Initial relation vocabulary:

- `left_of`
- `right_of`
- `above`
- `below`
- `overlap`
- `inside`
- `intersect`
- `larger_than`
- `smaller_than`

Distance and depth relations should be delayed until the pipeline is stable.

### 6.3 Positive and Negative Questions

For each valid object pair `(A, B)`:

- positive question: `Is A left of B?`
- negative question: `Is A right of B?`

Hard negatives are built by:

- swapping near-by same-class instances,
- substituting confusable relations,
- degrading one crop to create `insufficient` cases.

## 7. Stage-wise Training

### Stage 1: Pointer Pretraining

Train only evidence localization:

\[
\mathcal{L}^{(1)} = \mathcal{L}_{ptr}
\]

### Stage 2: Verifier Pretraining

Use GT crops and train only verification:

\[
\mathcal{L}^{(2)} =
\mathcal{L}_{ver} +
\lambda_{cf}\mathcal{L}_{cf} +
\lambda_{cons}\mathcal{L}_{cons}
\]

### Stage 3: Joint Fine-tuning

Connect the full loop:

\[
\mathcal{L}^{(3)} =
\mathcal{L}_{ans} +
\lambda_{final}\mathcal{L}_{final} +
\lambda_{ptr}\mathcal{L}_{ptr} +
\lambda_{ver}\mathcal{L}_{ver} +
\lambda_{cf}\mathcal{L}_{cf} +
\lambda_{cons}\mathcal{L}_{cons}
\]

## 8. Why This Is Different

AVV does not assume the model needs deeper latent iteration by default. It assumes a simpler failure mode:

- the answer was proposed before the relevant visual detail was checked.

This makes the intervention smaller than redesigning the entire encoder-decoder stack, but more targeted than adding generic reasoning tokens.

## 9. Practical Ablations

The first ablations that matter are:

1. no re-perception, verifier runs on first-pass features only
2. no verifier, direct answer from proposal
3. no counterfactual training
4. no consistency loss
5. GT evidence boxes vs predicted evidence boxes

These ablations test whether the gains come from actual visual verification rather than extra parameters.
