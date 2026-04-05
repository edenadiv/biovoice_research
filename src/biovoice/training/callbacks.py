"""Training callbacks such as early stopping."""

from __future__ import annotations


class EarlyStopping:
    """Simple early stopping utility keyed on validation loss."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        """Return ``True`` when training should stop."""
        if value < self.best:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
