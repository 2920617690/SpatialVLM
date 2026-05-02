# QCR: Query-Conditioned Re-encoding

## 1. Goal

This project is organized around one narrow question:

**does it help if the same image is encoded a second time after the model already knows the question?**

The point is not to start with crop policies, RL agents, or inspect controllers. The point is to isolate the smallest plausible mechanism.

## 2. Two Models

The repo compares:

1. `one-pass baseline`
2. `QCR two-pass`

### Baseline

```text
image -> ViT -> projector -> decoder -> answer
```

### QCR

```text
pass 1:
  image -> ViT -> projector -> decoder
  read hidden state at <reencode_slot>

pass 2:
  project that hidden state into a condition token
  prepend it to the same image token sequence
  run a second image encoding pass
  residual-refine first-pass visual tokens
  decode final answer
```

## 3. Forward Pass

Let:

- `P` be image patch tokens
- `G1` be first-pass visual tokens
- `h_q` be the decoder hidden state at `<reencode_slot>`
- `c` be the condition token
- `G2` be second-pass visual tokens
- `R` be refined visual tokens

### Pass 1

```text
P = PatchEmbed(x)
G1 = ViT(P)
h_q = Decoder(Project(G1), q + <reencode_slot>)
```

### Condition Token

```text
c = W_c(h_q)
```

### Pass 2

```text
G2_full = ViT([c ; P])
G2 = G2_full[:, 1:, :]
```

### Residual Fusion

```text
alpha = sigmoid(W_a(h_q))
R = G1 + alpha * (G2 - G1)
```

The final decoder sees `Project(R)` instead of `Project(G1)`.

## 4. Training

The synthetic dataset provides:

- `question`
- `draft_response`
- `final_response`

The baseline trainer uses `final_response` only.

The QCR trainer uses:

- optional pass-1 draft loss
- pass-2 final answer loss

```text
L = lambda_draft * L_draft + lambda_final * L_final
```

## 5. Why This Version First

QCR v0 deliberately avoids:

- crop policies
- box regression
- recurrent inspect loops
- RL control

Those may matter later, but they should not come before we know whether question-conditioned second-pass encoding helps at all.

## 6. Strict vs Fallback Backend

The intended method is:

```text
[condition token ; image patch tokens] -> same ViT blocks
```

However, different Qwen checkpoints expose different internals. The code therefore supports two backends:

1. `strict_shared_vit`
2. `projected_token_refiner` fallback

The fallback preserves the two-pass logic but applies the condition token in projected visual token space if direct vision-tower re-entry is unavailable.

## 7. First Ablations

The minimum comparison should be:

1. one-pass baseline
2. two-pass with dummy condition token
3. two-pass with question-conditioned token

That tells you whether:

- two-pass helps at all
- the gain is just extra depth or compute
- or the question-conditioned token really changes the visual encoding
