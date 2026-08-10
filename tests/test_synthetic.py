"""Tests for the synthetic dataset generator."""
from __future__ import annotations

import numpy as np
import torch

from moodsync.models.fusion import FusionMLP
from scripts.generate_synthetic import make_dataset


def test_shapes_and_dtypes():
    X, y_dist, y_mis = make_dataset(n=128, p_disagree=0.5, p_audio=0.7, seed=0)
    assert X.shape == (128, FusionMLP.INPUT_DIM)
    assert y_dist.shape == (128, 7)
    assert y_mis.shape == (128,)
    assert X.dtype == torch.float32
    assert y_dist.dtype == torch.float32
    assert y_mis.dtype == torch.float32


def test_target_distributions_normalise():
    _, y_dist, _ = make_dataset(n=64, p_disagree=0.5, p_audio=0.7, seed=0)
    sums = y_dist.sum(dim=-1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-4)


def test_mismatch_balance_respects_p_disagree():
    _, _, y_mis = make_dataset(n=2000, p_disagree=0.5, p_audio=0.7, seed=0)
    rate = y_mis.mean().item()
    assert 0.4 < rate < 0.6   # within ±10% of the requested 0.5


def test_audio_presence_matches_p_audio():
    X, _, _ = make_dataset(n=2000, p_disagree=0.5, p_audio=0.6, seed=0)
    audio_mask = X[:, FusionMLP.INPUT_DIM - 4]   # the audio mask bit
    rate = audio_mask.mean().item()
    assert 0.5 < rate < 0.7
