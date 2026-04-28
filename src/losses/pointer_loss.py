import torch
import torch.nn.functional as F


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    return widths * heights


def _giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(pred[:, 0], target[:, 0])
    y1 = torch.maximum(pred[:, 1], target[:, 1])
    x2 = torch.minimum(pred[:, 2], target[:, 2])
    y2 = torch.minimum(pred[:, 3], target[:, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)

    pred_area = _box_area(pred)
    target_area = _box_area(target)
    union = pred_area + target_area - inter + 1.0e-6
    iou = inter / union

    cx1 = torch.minimum(pred[:, 0], target[:, 0])
    cy1 = torch.minimum(pred[:, 1], target[:, 1])
    cx2 = torch.maximum(pred[:, 2], target[:, 2])
    cy2 = torch.maximum(pred[:, 3], target[:, 3])
    closure = (cx2 - cx1).clamp(min=1.0e-6) * (cy2 - cy1).clamp(min=1.0e-6)
    giou = iou - (closure - union) / closure
    return 1.0 - giou.mean()


def pointer_box_loss(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> torch.Tensor:
    l1 = F.l1_loss(pred_a, target_a) + F.l1_loss(pred_b, target_b)
    giou = _giou_loss(pred_a, target_a) + _giou_loss(pred_b, target_b)
    return l1 + giou
