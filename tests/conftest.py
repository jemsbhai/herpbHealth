"""Shared test fixtures for herpb_health."""

import pytest
import numpy as np


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_embeddings(rng):
    """Small set of normalized float32 embeddings (10 vectors, dim=384)."""
    vecs = rng.standard_normal((10, 384)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms
