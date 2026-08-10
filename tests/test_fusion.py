"""Tests for the fusion module.

Heuristic fusion is fully tested. Learned fusion is tested for shape and
fallback behaviour without requiring trained weights.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from moodsync.config import CANONICAL_EMOTIONS
from moodsync.models.fusion import (
    FusionMLP,
    LearnedFusion,
    heuristic_fuse,
)


# Helpers ------------------------------------------------------------- #
def _onehot(label: str) -> np.ndarray:
    v = np.zeros(7)
    v[CANONICAL_EMOTIONS.index(label)] = 1.0
    return v


def _peaked(label: str, peak: float = 0.7) -> np.ndarray:
    v = np.full(7, (1 - peak) / 6)
    v[CANONICAL_EMOTIONS.index(label)] = peak
    return v


# --------------------------------------------------------------------- #
class TestHeuristicAgreement:
    def test_agreement_no_mismatch(self):
        v = _peaked("happy")
        t = _peaked("happy")
        out = heuristic_fuse(vision=v, text=t)
        assert out.top_label == "happy"
        assert not out.mismatch
        assert out.strategy == "heuristic"

    def test_distribution_sums_to_one(self):
        out = heuristic_fuse(vision=_peaked("sad"), text=_peaked("sad"))
        assert abs(out.distribution.sum() - 1.0) < 1e-6


class TestHeuristicMismatch:
    def test_valence_conflict_triggers_mismatch(self):
        # The exact scenario from the brief: face=sad, text=happy.
        v = _peaked("sad", 0.7)
        t = _peaked("happy", 0.8)
        out = heuristic_fuse(vision=v, text=t)
        assert out.mismatch
        assert "valence" in out.mismatch_reason.lower()

    def test_neutral_with_negative_no_mismatch(self):
        # Neutral + sad: not a "positive vs negative" conflict.
        v = _peaked("neutral", 0.6)
        t = _peaked("sad", 0.6)
        out = heuristic_fuse(vision=v, text=t)
        # Whether mismatch fires depends on JS threshold; either way should not
        # report valence-conflict reasoning.
        assert "valence" not in out.mismatch_reason.lower()

    def test_three_modality_works(self):
        out = heuristic_fuse(
            vision=_peaked("happy"),
            text=_peaked("happy"),
            audio=_peaked("sad"),  # audio disagrees
        )
        assert out.distribution.shape == (7,)
        assert "audio" in out.modality_distributions


class TestSingleModality:
    def test_single_modality_no_mismatch(self):
        out = heuristic_fuse(vision=_peaked("angry"))
        assert not out.mismatch
        assert "single modality" in out.mismatch_reason.lower()

    def test_no_modality_raises(self):
        with pytest.raises(ValueError):
            heuristic_fuse()


# --------------------------------------------------------------------- #
class TestFusionMLP:
    def test_input_shape(self):
        x = FusionMLP.make_input(_peaked("happy"), _peaked("sad"))
        assert x.shape == (FusionMLP.INPUT_DIM,)

    def test_input_with_audio(self):
        x = FusionMLP.make_input(_peaked("happy"), _peaked("sad"), audio=_peaked("neutral"))
        assert x.shape == (FusionMLP.INPUT_DIM,)
        # Audio mask bit should be 1.0
        assert x[7 + 7 + 7].item() == 1.0

    def test_input_without_audio_mask_zero(self):
        x = FusionMLP.make_input(_peaked("happy"), _peaked("sad"))
        assert x[7 + 7 + 7].item() == 0.0

    def test_forward_shape(self):
        m = FusionMLP()
        x = FusionMLP.make_input(_peaked("happy"), _peaked("sad")).unsqueeze(0)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, FusionMLP.OUTPUT_DIM)


class TestLearnedFusion:
    def test_falls_back_when_weights_missing(self, tmp_path):
        bogus_path = tmp_path / "no_such_file.pt"
        lf = LearnedFusion(weights_path=bogus_path)
        assert not lf.is_available
        out = lf.fuse(vision=_peaked("happy"), text=_peaked("sad"))
        # Should fall back to heuristic — strategy field reveals it.
        assert out.strategy == "heuristic"
