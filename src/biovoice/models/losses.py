"""Loss functions used by the baseline models.

The repository keeps the branch baselines intentionally compact, but the spoof
branch still needs a little flexibility because ASVspoof-style corpora can be
quite skewed. A weighted BCE option and a focal-loss option are both small,
well-understood changes that help us test whether weak spoof performance is
primarily a thresholding issue or an optimization issue.
"""

from __future__ import annotations

import torch
from torch import nn


speaker_classification_loss = nn.CrossEntropyLoss()
spoof_classification_loss = nn.BCEWithLogitsLoss()
fusion_classification_loss = nn.CrossEntropyLoss()


def build_spoof_loss(
    *,
    loss_name: str = "bce",
    pos_weight: float | None = None,
    focal_gamma: float = 2.0,
) -> callable:
    """Build the configured spoof loss function.

    Parameters
    ----------
    loss_name:
        ``bce`` keeps the previous baseline behavior.
        ``weighted_bce`` adds a positive-class weight.
        ``focal_bce`` applies a simple sigmoid focal loss around BCE.
    pos_weight:
        Positive-class weight used by ``weighted_bce`` and ``focal_bce`` when
        provided. It should usually be derived from the train split only.
    focal_gamma:
        Down-weights very easy examples when focal loss is enabled.
    """
    normalized_name = str(loss_name).lower()
    if normalized_name == "bce":
        base_loss = nn.BCEWithLogitsLoss()
        return lambda logits, labels: base_loss(logits, labels)
    if normalized_name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("weighted_bce requires a positive-class weight.")
        def _weighted_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=logits.device),
            )

        return _weighted_loss
    if normalized_name == "focal_bce":
        if pos_weight is None:
            raise ValueError("focal_bce requires a positive-class weight.")

        def _focal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
                pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=logits.device),
            )
            probabilities = torch.sigmoid(logits)
            pt = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
            focal_weight = torch.pow(1.0 - pt, float(focal_gamma))
            return (focal_weight * bce).mean()

        return _focal_loss
    raise ValueError("Unsupported spoof loss. Expected one of: bce, weighted_bce, focal_bce.")
