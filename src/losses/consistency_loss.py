import torch
import torch.nn.functional as F


def consistency_kl_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_probs_a = F.log_softmax(logits_a, dim=-1)
    probs_b = F.softmax(logits_b, dim=-1)
    return F.kl_div(log_probs_a, probs_b, reduction="batchmean")
