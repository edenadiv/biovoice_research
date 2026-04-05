"""Unit tests for enrollment aggregation."""

from __future__ import annotations

import torch

from biovoice.data.enrollment import aggregate_embeddings


def test_enrollment_aggregation_returns_embedding_vector() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    aggregate = aggregate_embeddings(embeddings)
    assert aggregate.shape == (2,)
