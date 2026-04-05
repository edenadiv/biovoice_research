"""Loss functions used by the baseline models."""

from __future__ import annotations

from torch import nn


speaker_classification_loss = nn.CrossEntropyLoss()
spoof_classification_loss = nn.BCEWithLogitsLoss()
fusion_classification_loss = nn.CrossEntropyLoss()
