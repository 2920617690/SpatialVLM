# AVV with Qwen3.5-4B

## 1. Concrete Base Model

This codebase assumes a real multimodal base model:

```text
image -> ViT -> projector -> decoder-only LLM
```

The default implementation targets `Qwen3.5-4B`.

The working claim is not that we need more permanent reasoning modules. The working claim is that the same VLM should learn a better control policy over answering:

```text
draft -> verify -> final
```

## 2. Training Objective

We train the shared Qwen model in three supervised modes:

### Draft Mode

```text
input: image + question
target: structured draft response
```

The target contains:

- proposed answer
- draft claim
- decomposed subclaims

### Verify Mode

```text
input: image + question + draft claim
target: structured verification response
```

The target contains:

- per-subclaim verdicts
- overall verdict in `{support, contradict, insufficient}`

### Final Mode

```text
input: image + question + draft response + verification response
target: final answer
```

## 3. Policy Learning

After warm start, we treat the same shared model as a policy over high-level actions:

```text
PROPOSE
VERIFY
REVISE
ANSWER
ABSTAIN
```

The main learning target is not "how to do vision from scratch." It is:

- when to verify
- when to revise
- when to stop

## 4. BVPO

The repo includes a lightweight stage-2 trainer called `BVPO`:

`Budgeted Verification Policy Optimization`

Its reward has four components:

```text
R =
  answer_reward
  + verify_reward
  - step_cost
  - contradiction_penalty
```

The purpose of the algorithm is to make verification useful but not free. If the model keeps verifying without need, return should go down.

## 5. Why Synthetic Data Comes First

Before using noisy real-world QA, the repo uses synthetic spatial scenes because they provide:

- exact object identities
- exact bounding boxes
- exact subclaims
- exact verification labels
- exact imitation trajectories

This is especially important for training `verify mode` and stage-1 imitation.

## 6. Sample Structure

Each training sample stores:

```text
image_path
question
answer
draft_response
verify_response
final_response
subclaims
trajectory
scene_objects
```

This structure supports:

- stage-0 SFT
- stage-1 imitation
- stage-2 RL rollouts

## 7. Recommended Experimental Order

The intended experimental order is:

1. one-shot baseline on synthetic tasks
2. stage-0 SFT on `draft / verify / final`
3. stage-1 imitation on oracle trajectories
4. stage-2 BVPO fine-tuning
5. later transfer to real data

That order matters. If stage-0 and stage-1 do not help, RL should not be trusted to rescue the idea.
