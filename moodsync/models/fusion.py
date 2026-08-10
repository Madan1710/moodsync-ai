"""Multimodal fusion + mismatch detection.

Two fusion strategies are implemented side-by-side:

  1. **Heuristic fusion** (baseline). A weighted average of canonical 7-vectors
     plus a rules-based mismatch flag from valence groups + KL divergence.
     Always available — no training required.

  2. **Learned fusion** (extended feature). A small MLP takes the concatenated
     canonical distributions plus disagreement features and outputs a fused
     distribution + a mismatch logit. Trained on a synthetic dataset where we
     control the ground truth of agree/disagree (see
     `scripts/generate_synthetic.py` and `scripts/train_fusion.py`).

The mismatch logic is the project's core insight ("face shows sadness while
words say I'm fine") — both strategies surface it explicitly and the UI
shows an amber badge when triggered.

Why have both? Examiners want to see the baseline → improvement story, and
the heuristic falls back gracefully if trained weights are missing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moodsync.config import (
    CANONICAL_EMOTIONS,
    FUSION_CONFIG,
    PATH_CONFIG,
)
from moodsync.utils.alignment import (
    jensen_shannon,
    kl_divergence,
    top_k,
    valence_distribution,
    valence_of,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #
@dataclass
class FusionResult:
    """Output of the fusion layer (heuristic or learned)."""

    distribution: np.ndarray                       # canonical 7-vector
    top_label: str
    top_confidence: float
    mismatch: bool
    mismatch_score: float                          # JS divergence in [0, ~0.7]
    mismatch_reason: str                           # human-readable explanation
    valence: Dict[str, float]                      # positive/negative/neutral mass
    modality_distributions: Dict[str, np.ndarray]  # for downstream UI/explainer
    strategy: str                                   # "heuristic" or "learned"


# --------------------------------------------------------------------------- #
#  Heuristic fusion
# --------------------------------------------------------------------------- #
def heuristic_fuse(
    vision: Optional[np.ndarray] = None,
    text: Optional[np.ndarray] = None,
    audio: Optional[np.ndarray] = None,
) -> FusionResult:
    """Weighted-average fusion + valence/KL mismatch rule.

    Any subset of modalities may be None; weights are renormalised over what's
    present so the demo still works with just an image, just text, etc.
    """
    modalities = _collect_modalities(vision, text, audio)
    if not modalities:
        raise ValueError("heuristic_fuse requires at least one modality")

    weights = _resolve_weights(modalities.keys())
    fused = sum(modalities[name] * weights[name] for name in modalities)
    fused = _renormalise(fused)

    label, conf = top_k(fused, k=1)[0]
    val = valence_distribution(fused)

    mismatch, score, reason = _detect_mismatch(modalities)

    return FusionResult(
        distribution=fused,
        top_label=label,
        top_confidence=conf,
        mismatch=mismatch,
        mismatch_score=score,
        mismatch_reason=reason,
        valence=val,
        modality_distributions=modalities,
        strategy="heuristic",
    )


def _collect_modalities(
    vision: Optional[np.ndarray],
    text: Optional[np.ndarray],
    audio: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    out = {}
    if vision is not None:
        out["vision"] = np.asarray(vision, dtype=np.float64)
    if text is not None:
        out["text"] = np.asarray(text, dtype=np.float64)
    if audio is not None:
        out["audio"] = np.asarray(audio, dtype=np.float64)
    return out


def _resolve_weights(present: "set | list") -> Dict[str, float]:
    """Pull weights from config, then renormalise over present modalities only."""
    base = {
        "vision": FUSION_CONFIG.weight_vision,
        "text": FUSION_CONFIG.weight_text,
        "audio": FUSION_CONFIG.weight_audio if "audio" in present else 0.0,
    }
    # If audio is present but its config weight is 0, give it a third of the mass.
    if "audio" in present and base["audio"] == 0.0:
        base["audio"] = 1.0 / 3.0
        base["vision"] *= 2.0 / 3.0
        base["text"] *= 2.0 / 3.0

    filtered = {k: v for k, v in base.items() if k in present}
    total = sum(filtered.values())
    if total <= 0:
        return {k: 1.0 / len(filtered) for k in filtered}
    return {k: v / total for k, v in filtered.items()}


def _renormalise(p: np.ndarray) -> np.ndarray:
    p = np.maximum(p, 0.0)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))


def _detect_mismatch(modalities: Dict[str, np.ndarray]):
    """Return (mismatch_bool, score, reason)."""
    if len(modalities) < 2:
        return False, 0.0, "Single modality — no cross-check possible."

    # Pairwise JS over all modality pairs; the worst pair dominates.
    names = list(modalities.keys())
    worst_pair = None
    worst_js = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            js = jensen_shannon(modalities[names[i]], modalities[names[j]])
            if js > worst_js:
                worst_js = js
                worst_pair = (names[i], names[j])

    # Valence-conflict override: if any two modalities' top-1 fall into
    # opposing valence groups (positive vs negative), always flag.
    valence_conflict = False
    valence_pair = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            top_i = CANONICAL_EMOTIONS[int(np.argmax(modalities[names[i]]))]
            top_j = CANONICAL_EMOTIONS[int(np.argmax(modalities[names[j]]))]
            v_i = valence_of(top_i)
            v_j = valence_of(top_j)
            if {v_i, v_j} == {"positive", "negative"}:
                valence_conflict = True
                valence_pair = (names[i], top_i, names[j], top_j)
                break

    if valence_conflict and FUSION_CONFIG.valence_mismatch_overrides_kl:
        a, ta, b, tb = valence_pair
        reason = (
            f"Valence conflict: {a} reads '{ta}' (negative) vs "
            f"{b} reads '{tb}' (positive)."
        )
        return True, max(worst_js, FUSION_CONFIG.kl_mismatch_threshold + 0.01), reason

    if worst_js >= FUSION_CONFIG.kl_mismatch_threshold:
        a, b = worst_pair
        reason = (
            f"High distributional divergence between {a} and {b} "
            f"(JS={worst_js:.2f} ≥ {FUSION_CONFIG.kl_mismatch_threshold:.2f})."
        )
        return True, worst_js, reason

    return False, worst_js, "Modalities are in agreement."


# --------------------------------------------------------------------------- #
#  Learned fusion (small MLP)
# --------------------------------------------------------------------------- #
class FusionMLP(nn.Module):
    """Small MLP: [vision | text | audio | disagreement-features] -> fused dist + mismatch logit.

    Audio is optional — when absent we feed zeros and a flag bit. This keeps
    the network shape constant so we can train once and use it for any
    modality combination.

    Architecture (deliberately small for fast CPU inference):

        in_dim = 7+7+7 (modality probs) + 1 (audio mask) + 3 (disagree feats)
              = 25
        Linear(25, 64) -> ReLU -> Dropout(0.3)
        Linear(64, 32) -> ReLU
        Linear(32, 8)   # 7 fused logits + 1 mismatch logit
    """

    INPUT_DIM = 7 + 7 + 7 + 1 + 3
    OUTPUT_DIM = 7 + 1

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def make_input(
        vision: np.ndarray,
        text: np.ndarray,
        audio: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Pack modality vectors + disagreement features into the network input."""
        v = np.asarray(vision, dtype=np.float32)
        t = np.asarray(text, dtype=np.float32)
        if audio is not None:
            a = np.asarray(audio, dtype=np.float32)
            audio_mask = 1.0
        else:
            a = np.zeros(7, dtype=np.float32)
            audio_mask = 0.0

        # Disagreement features:
        #   * JS(v, t)
        #   * cosine distance(v, t)
        #   * abs difference of top-1 confidences
        js = float(jensen_shannon(v, t))
        cos = 1.0 - float(np.dot(v, t) / (np.linalg.norm(v) * np.linalg.norm(t) + 1e-9))
        conf_diff = float(abs(v.max() - t.max()))
        feats = np.array([js, cos, conf_diff], dtype=np.float32)

        # Cast the audio mask scalar to float32 too — without this, the np.concatenate
        # promotes the whole vector to float64 and torch.nn.Linear (float32 weights)
        # raises a dtype mismatch at forward time.
        mask = np.array([audio_mask], dtype=np.float32)
        packed = np.concatenate([v, t, a, mask, feats]).astype(np.float32)
        return torch.from_numpy(packed)


class LearnedFusion:
    """Wrapper that loads trained MLP weights (or falls back to heuristic)."""

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        self._weights_path = weights_path or PATH_CONFIG.fusion_weights
        self._model: Optional[FusionMLP] = None

    @property
    def is_available(self) -> bool:
        return self._weights_path.exists()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.is_available:
            return False
        try:
            model = FusionMLP()
            state = torch.load(self._weights_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self._model = model
            return True
        except Exception as e:  # pragma: no cover
            logger.warning("Could not load learned fusion weights: %s", e)
            return False

    def fuse(
        self,
        vision: np.ndarray,
        text: np.ndarray,
        audio: Optional[np.ndarray] = None,
    ) -> FusionResult:
        """Run the learned MLP. Falls back to heuristic if weights missing."""
        if not self._ensure_loaded():
            logger.info("Learned fusion weights unavailable; falling back to heuristic.")
            return heuristic_fuse(vision=vision, text=text, audio=audio)

        x = FusionMLP.make_input(vision, text, audio).unsqueeze(0)
        with torch.no_grad():
            out = self._model(x)[0].numpy()

        fused_logits = out[:7]
        mismatch_logit = out[7]

        fused = _renormalise(np.exp(fused_logits - fused_logits.max()))
        mismatch_prob = 1.0 / (1.0 + np.exp(-mismatch_logit))
        mismatch = bool(mismatch_prob >= 0.5)

        modalities = _collect_modalities(vision, text, audio)
        # Reuse heuristic explanation for transparency even when learned says yes.
        _, h_score, h_reason = _detect_mismatch(modalities)

        label, conf = top_k(fused, k=1)[0]
        return FusionResult(
            distribution=fused,
            top_label=label,
            top_confidence=conf,
            mismatch=mismatch,
            mismatch_score=float(mismatch_prob),
            mismatch_reason=(
                f"Learned fusion p(mismatch)={mismatch_prob:.2f}. " + h_reason
            ),
            valence=valence_distribution(fused),
            modality_distributions=modalities,
            strategy="learned",
        )


def get_learned_fusion() -> LearnedFusion:
    return LearnedFusion()


__all__ = [
    "FusionResult",
    "heuristic_fuse",
    "FusionMLP",
    "LearnedFusion",
    "get_learned_fusion",
]
