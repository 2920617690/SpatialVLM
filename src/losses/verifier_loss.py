import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def counterfactual_contradiction_loss(
    logits: torch.Tensor,
    contradict_index: int = 1,
) -> torch.Tensor:
    target = torch.full(
        (logits.shape[0],),
        contradict_index,
        dtype=torch.long,
        device=logits.device,
    )
    return F.cross_entropy(logits, target)
