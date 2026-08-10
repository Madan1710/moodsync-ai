"""End-to-end integration test: synthesise some realistic-shaped probability
vectors and confirm the *trained* learned fusion correctly distinguishes
agreement from valence-conflict mismatches.

This doesn't require any HF model downloads — it just checks the trained
fusion network is plugged in correctly and the data flow holds together.
"""
from __future__ import annotations

import numpy as np
import pytest

from moodsync.config import CANONICAL_EMOTIONS, PATH_CONFIG
from moodsync.models.fusion import LearnedFusion, heuristic_fuse


def _peaked(label: str, peak: float = 0.75) -> np.ndarray:
    v = np.full(7, (1 - peak) / 6)
    v[CANONICAL_EMOTIONS.index(label)] = peak
    return v


@pytest.mark.skipif(
    not PATH_CONFIG.fusion_weights.exists(),
    reason="Trained fusion weights not present — run scripts.train_fusion first.",
)
class TestLearnedFusionIntegration:
    def test_agreement_predicts_no_mismatch(self):
        lf = LearnedFusion()
        out = lf.fuse(vision=_peaked("happy"), text=_peaked("happy"))
        assert out.strategy == "learned"
        # Trained MLP should put low mismatch prob on agreement.
        assert not out.mismatch
        assert out.mismatch_score < 0.5
        assert out.top_label == "happy"

    def test_valence_conflict_predicts_mismatch(self):
        # Brief's example: face=sad, text=happy.
        lf = LearnedFusion()
        out = lf.fuse(vision=_peaked("sad"), text=_peaked("happy"))
        assert out.strategy == "learned"
        assert out.mismatch
        assert out.mismatch_score >= 0.5

    def test_three_modality_runs(self):
        lf = LearnedFusion()
        out = lf.fuse(
            vision=_peaked("happy"),
            text=_peaked("happy"),
            audio=_peaked("happy"),
        )
        assert out.distribution.shape == (7,)
        assert out.top_label == "happy"
