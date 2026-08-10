"""Tests for emotion label alignment helpers.

These are deliberately fast (no model loading) so they run in <1s and
can be wired into CI cheaply.
"""
from __future__ import annotations

import numpy as np
import pytest

from moodsync.config import CANONICAL_EMOTIONS
from moodsync.utils.alignment import (
    emojify,
    jensen_shannon,
    kl_divergence,
    normalize_label,
    softmax_with_temperature,
    to_canonical_distribution,
    top_k,
    valence_distribution,
    valence_of,
)


class TestNormalizeLabel:
    def test_canonical_passes_through(self):
        for emo in CANONICAL_EMOTIONS:
            assert normalize_label(emo) == emo

    def test_aliases_map_correctly(self):
        assert normalize_label("anger") == "angry"
        assert normalize_label("joy") == "happy"
        assert normalize_label("sadness") == "sad"

    def test_case_and_whitespace(self):
        assert normalize_label("  HAPPY  ") == "happy"
        assert normalize_label("Joy") == "happy"


class TestToCanonical:
    def test_text_model_aliases(self):
        # Simulate j-hartmann/emotion-english-distilroberta-base output.
        labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        probs = [0.0, 0.0, 0.0, 0.9, 0.05, 0.05, 0.0]
        out = to_canonical_distribution(labels, probs)
        assert out[CANONICAL_EMOTIONS.index("happy")] == pytest.approx(0.9)
        assert out[CANONICAL_EMOTIONS.index("sad")] == pytest.approx(0.05)
        assert np.isclose(out.sum(), 1.0)

    def test_audio_partial_classes(self):
        # HuBERT-superb-er has only 4 classes.
        labels = ["ang", "hap", "neu", "sad"]
        probs = [0.1, 0.1, 0.1, 0.7]
        out = to_canonical_distribution(labels, probs)
        assert out[CANONICAL_EMOTIONS.index("sad")] == pytest.approx(0.7)
        # disgust/fear/surprise stay at zero.
        assert out[CANONICAL_EMOTIONS.index("disgust")] == 0.0

    def test_empty_input_uniform_fallback(self):
        out = to_canonical_distribution(["totally_made_up"], [1.0])
        assert np.allclose(out, np.full(7, 1 / 7))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            to_canonical_distribution(["happy"], [0.5, 0.5])


class TestSoftmaxTemperature:
    def test_default_matches_softmax(self):
        logits = np.array([2.0, 1.0, 0.5])
        result = softmax_with_temperature(logits, temperature=1.0)
        expected = np.exp(logits) / np.exp(logits).sum()
        assert np.allclose(result, expected)

    def test_high_temperature_smooths(self):
        logits = np.array([5.0, 0.0, 0.0])
        cold = softmax_with_temperature(logits, temperature=0.5)
        hot = softmax_with_temperature(logits, temperature=2.0)
        # Hot should be more uniform.
        assert hot.max() < cold.max()
        assert np.isclose(hot.sum(), 1.0)
        assert np.isclose(cold.sum(), 1.0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            softmax_with_temperature(np.array([1.0]), temperature=0.0)


class TestKLAndJS:
    def test_identical_distributions_zero_divergence(self):
        p = np.array([0.5, 0.5])
        assert kl_divergence(p, p) < 1e-6
        assert jensen_shannon(p, p) < 1e-6

    def test_disjoint_distributions_high_js(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert jensen_shannon(p, q) > 0.6  # max JS is ln(2)/2 ≈ 0.347 in nats? actually 0.693 here

    def test_js_is_symmetric(self):
        p = np.array([0.7, 0.3])
        q = np.array([0.4, 0.6])
        assert abs(jensen_shannon(p, q) - jensen_shannon(q, p)) < 1e-9


class TestTopK:
    def test_returns_correct_order(self):
        d = np.array([0.05, 0.1, 0.05, 0.6, 0.1, 0.05, 0.05])
        top = top_k(d, k=2)
        assert top[0][0] == "happy"
        assert top[0][1] == pytest.approx(0.6)
        assert top[1][1] == pytest.approx(0.1)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            top_k(np.zeros(5), k=1)


class TestValence:
    def test_valence_of_each_class(self):
        assert valence_of("happy") == "positive"
        assert valence_of("sad") == "negative"
        assert valence_of("neutral") == "neutral"
        assert valence_of("joy") == "positive"  # alias

    def test_valence_distribution_sums_to_one(self):
        d = np.full(7, 1 / 7)
        v = valence_distribution(d)
        assert abs(sum(v.values()) - 1.0) < 1e-9


class TestEmojify:
    def test_known_labels(self):
        assert emojify("happy") == "😄"
        assert emojify("sad") == "😢"

    def test_unknown_returns_fallback(self):
        assert emojify("nonsense") == "❔"
