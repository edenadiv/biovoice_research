"""Training callbacks such as early stopping."""

from __future__ import annotations


class EarlyStopping:
    """Simple early stopping utility keyed on validation loss."""

    def __init__(self, patience: int, mode: str = "min") -> None:
        self.patience = patience
        if mode not in {"min", "max"}:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'.")
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        """Return ``True`` when training should stop."""
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
