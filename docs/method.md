# AVV: Self-Verification Policy Learning for VLMs

## 1. Setting

We consider the standard VLM setting:

```text
image -> ViT -> projector -> decoder-only LLM -> answer
```

The central claim of this project is that many spatial errors do not come from missing model capacity alone. They come from **premature commitment**. The model proposes an answer before it has verified whether the underlying visual claim is actually supported by the image.

Instead of redesigning the whole architecture, we keep the original `ViT + projector + decoder` backbone and change the problem from one-shot answering to multi-step self-verification.

## 2. Core Hypothesis

For spatial questions, the model should not only answer. It should learn a policy that decides:

1. when to propose a draft answer
2. when to verify that draft
3. whether to revise the draft after verification
4. when to stop and commit

This is better viewed as a sequential decision problem than as a single supervised mapping.

## 3. Why This Is Not Just Prompt Engineering

If we only ask the same VLM to "think twice" at inference time, the method is just prompt engineering.

It becomes a real method only when the model is trained to perform three distinct behaviors:

1. `draft mode`: produce a draft answer or claim
2. `verify mode`: judge whether the draft is supported by the image
3. `final mode`: produce the final answer after incorporating the verification result

These are not just different prompts. They are different supervised or reinforced behaviors over shared model parameters.

## 4. Shared-Backbone Formulation

We intentionally avoid assuming new permanent modules. The preferred formulation reuses the same VLM across multiple passes.

### 4.1 Draft Pass

Given image `x` and question `q`, the decoder produces:

```text
y0 = draft answer
c0 = draft claim
```

Example:

```text
Question: Is the cat to the left of the dog?
Draft: answer = yes, claim = cat left of dog
```

### 4.2 Verify Pass

The same VLM is then called again in verification mode:

```text
input: image x + claim c0
output: v0 in {support, contradict, insufficient}
```

This is deliberately a narrow output space. The goal is to reduce shortcut freedom and force a hard judgment.

### 4.3 Final Pass

The model is called one more time:

```text
input: image x + question q + draft y0/c0 + verification result v0
output: final answer y*
```

The final answer may preserve, revise, or reject the initial draft.

## 5. RL Formulation

We model self-verification as a partially observable sequential decision process.

### 5.1 State

The policy state at step `t` contains:

- image `x`
- question `q`
- current draft answer `y_t`
- current claim `c_t`
- verification history `h_t`
- remaining computation budget `B_t`

The state is not the full world state. It is the model's current belief and interaction history.

### 5.2 Action Space

We use a small discrete action space:

```text
PROPOSE
VERIFY
REVISE
ANSWER
ABSTAIN
```

Possible future refinements may add actions such as region-specific reinspection, but the minimal formulation should start with high-level control actions.

### 5.3 Transition

Each action triggers another pass through the same VLM in a different mode:

- `PROPOSE`: create or update a draft claim
- `VERIFY`: check whether the current claim is supported
- `REVISE`: rewrite the claim or answer after contradiction
- `ANSWER`: terminate and emit final answer
- `ABSTAIN`: terminate without committing to a risky answer

### 5.4 Reward

A practical reward should combine correctness and efficiency.

```text
R =
  final answer reward
  + verification consistency reward
  - unnecessary verification cost
  - contradiction-ignored penalty
```

A simple version is:

- correct final answer: `+1.0`
- correct verification label: `+0.2`
- each extra verification step: `-0.05`
- answering despite contradiction: `-0.3`
- abstention when truly ambiguous: small positive reward

The exact coefficients are secondary. The principle is that verification should help, but endless checking should be discouraged.

## 6. Training Pipeline

Pure RL from scratch is not the right starting point. The recommended pipeline has three stages.

### Stage 0: Supervised Warm Start

Train the shared VLM on three supervised formats:

#### Draft Mode

```text
input: image + question
target: draft answer + draft claim
```

#### Verify Mode

```text
input: image + claim
target: support / contradict / insufficient
```

#### Final Mode

```text
input: image + question + draft + verification
target: final answer
```

Loss:

```text
L_sft =
  L_draft
  + lambda_verify * L_verify
  + lambda_final * L_final
```

### Stage 1: Oracle-Guided Imitation

Use synthetic data or box-derived relations to construct good trajectories:

- if the relation is obvious, answer directly
- if the relation is hard or counterfactual, verify first
- if verification contradicts the draft, revise
- then answer

This stage teaches the policy a reasonable initial control flow before RL.

### Stage 2: RL Fine-tuning

Run policy optimization over multi-step episodes.

The objective is no longer just token imitation. It is expected return:

```text
J(pi) = E_pi [ sum_t gamma^t r_t ]
```

The exact algorithm is flexible. PPO-style fine-tuning is a natural starting point, but the method does not depend on one specific optimizer.

## 7. Data Construction

The warm-start and imitation stages still require structured supervision.

Each sample should be promoted from:

```text
(image, question, answer)
```

to:

```text
(image, question, claim, verify_label, final_answer)
```

Primary sources:

- COCO instances
- Visual Genome
- RefCOCO / RefCOCOg
- CLEVR
- synthetic grid, chart, table, or GUI tasks

For bbox-based datasets, relation labels can be programmatically derived:

- `left_of`
- `right_of`
- `above`
- `below`
- `overlap`
- `inside`
- `larger_than`
- `smaller_than`

This data is especially useful for training verify-mode and for imitation trajectories.

## 8. Why RL Helps

Supervised learning is good at teaching:

- what a valid draft looks like
- what a valid verification label looks like
- how to verbalize the final answer

RL is useful for what supervision does not naturally solve:

- when verification is necessary
- how often to verify
- when to revise
- when to stop

So RL should not replace the whole VLM training recipe. It should focus on **verification policy learning**.

## 9. Main Risk

If verification is judged only by the model itself, reward hacking becomes likely.

The model may learn to:

- invent a draft
- invent a matching verification label
- remain self-consistent without becoming visually faithful

Therefore, early training and reward design should rely as much as possible on external truth:

- synthetic geometry
- box-derived relations
- deterministic flip consistency
- programmatic relation checkers

## 10. Practical Ablations

The first ablations that matter are:

1. one-shot answer only
2. draft + final without verify mode
3. supervised multi-pass without RL
4. RL with no verification cost
5. RL with no contradiction penalty
6. GT relation labels vs noisy relation labels

These ablations test whether gains come from true policy learning rather than just longer prompting.
