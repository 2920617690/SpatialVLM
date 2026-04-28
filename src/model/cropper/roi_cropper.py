from __future__ import annotations

import torch
import torch.nn.functional as F


class ROICropper:
    def __init__(self, crop_size: int = 160) -> None:
        self.crop_size = crop_size

    def _crop_single(self, image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        x1, y1, x2, y2 = box.tolist()
        left = max(0, min(width - 1, int(x1 * width)))
        top = max(0, min(height - 1, int(y1 * height)))
        right = max(left + 1, min(width, int(x2 * width)))
        bottom = max(top + 1, min(height, int(y2 * height)))
        crop = image[:, top:bottom, left:right]
        return F.interpolate(
            crop.unsqueeze(0),
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def __call__(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        crops = [self._crop_single(image, box) for image, box in zip(images, boxes)]
        return torch.stack(crops, dim=0)
