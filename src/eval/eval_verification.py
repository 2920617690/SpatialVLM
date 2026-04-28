import torch


def verifier_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean().item()
