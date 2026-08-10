"""Generate a synthetic dataset for training the learned-fusion MLP.

Why synthetic? The brief gives us models for vision, text, and audio, but no
labelled multimodal corpus where ground-truth "the modalities agree" is
known. We construct one by:

  1. Sample a *true emotion* y ~ Uniform({happy, sad, ...}).
  2. With probability p_disagree, replace one or two modality predictions
     with a different emotion drawn from a contradicting valence group.
  3. Otherwise, all modality predictions cluster around y.
  4. Each modality's per-class probs are drawn from a Dirichlet centred on
     a one-hot vector for its assigned label, with concentration = α.

This gives us a controlled distribution we can train on — and crucially,
the ground-truth `mismatch` label is exact (we know whether the modalities
disagree because we constructed the disagreement).

Usage:
    python -m scripts.generate_synthetic --n 20000 --out assets/synthetic_fusion_data.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from moodsync.config import (
    CANONICAL_EMOTIONS,
    PATH_CONFIG,
    VALENCE_NEGATIVE,
    VALENCE_POSITIVE,
)
from moodsync.utils.alignment import jensen_shannon

NEG_IDX = [CANONICAL_EMOTIONS.index(e) for e in VALENCE_NEGATIVE]
POS_IDX = [CANONICAL_EMOTIONS.index(e) for e in VALENCE_POSITIVE]
NEU_IDX = CANONICAL_EMOTIONS.index("neutral")


def _dirichlet_around(label_idx: int, n_classes: int = 7, peak: float = 6.0, base: float = 0.6) -> np.ndarray:
    alpha = np.full(n_classes, base, dtype=np.float64)
    alpha[label_idx] += peak
    p = np.random.dirichlet(alpha)
    return p


def _opposing(label_idx: int) -> int:
    """Return an emotion in the opposing valence group."""
    if label_idx in POS_IDX:
        return int(np.random.choice(NEG_IDX))
    if label_idx in NEG_IDX:
        return int(np.random.choice(POS_IDX))
    # neutral: flip to either pos or neg
    return int(np.random.choice(POS_IDX + NEG_IDX))


def _disagreement_features(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    js = jensen_shannon(v, t)
    cos = 1.0 - float(np.dot(v, t) / (np.linalg.norm(v) * np.linalg.norm(t) + 1e-9))
    conf = float(abs(v.max() - t.max()))
    return np.array([js, cos, conf], dtype=np.float32)


def make_dataset(n: int, p_disagree: float = 0.5, p_audio: float = 0.7, seed: int = 42):
    """Build (X, y_dist, y_mismatch) tensors."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    n_classes = len(CANONICAL_EMOTIONS)
    X = np.zeros((n, 7 + 7 + 7 + 1 + 3), dtype=np.float32)
    y_dist = np.zeros((n, n_classes), dtype=np.float32)
    y_mis = np.zeros((n,), dtype=np.float32)

    for i in range(n):
        true_idx = int(rng.integers(0, n_classes))
        disagree = rng.random() < p_disagree

        if disagree:
            # Flip exactly one modality (vision OR text) with a strong opposing prediction.
            modality_to_flip = rng.choice(["vision", "text", "both"], p=[0.45, 0.45, 0.10])
            v_idx = _opposing(true_idx) if modality_to_flip in ("vision", "both") else true_idx
            t_idx = _opposing(true_idx) if modality_to_flip in ("text", "both") else true_idx
        else:
            v_idx = true_idx
            t_idx = true_idx

        v = _dirichlet_around(v_idx)
        t = _dirichlet_around(t_idx)

        has_audio = rng.random() < p_audio
        if has_audio:
            # Audio agrees with the modality majority (if any) more often than not.
            a_idx = true_idx if not disagree else int(rng.choice([v_idx, t_idx]))
            a = _dirichlet_around(a_idx)
            audio_mask = 1.0
        else:
            a = np.zeros(n_classes, dtype=np.float64)
            audio_mask = 0.0

        feats = _disagreement_features(v, t)
        X[i] = np.concatenate([v, t, a, [audio_mask], feats]).astype(np.float32)

        # Target distribution: average of modalities present, plus a slight
        # bias toward the *true* emotion when there's no disagreement.
        present = [v, t] + ([a] if has_audio else [])
        avg = np.mean(np.stack(present, axis=0), axis=0)
        if not disagree:
            avg[true_idx] += 0.1
            avg /= avg.sum()
        y_dist[i] = avg.astype(np.float32)
        y_mis[i] = 1.0 if disagree else 0.0

    return torch.from_numpy(X), torch.from_numpy(y_dist), torch.from_numpy(y_mis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--p-disagree", type=float, default=0.5)
    parser.add_argument("--p-audio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=PATH_CONFIG.synthetic_data)
    args = parser.parse_args()

    X, y_dist, y_mis = make_dataset(
        n=args.n, p_disagree=args.p_disagree, p_audio=args.p_audio, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "y_dist": y_dist, "y_mismatch": y_mis}, args.out)
    print(f"Saved {args.n:,} samples to {args.out}")
    print(f"  X shape:           {tuple(X.shape)}")
    print(f"  y_dist shape:      {tuple(y_dist.shape)}")
    print(f"  y_mismatch shape:  {tuple(y_mis.shape)}")
    print(f"  Class balance:     mismatch={y_mis.mean().item():.2%}")


if __name__ == "__main__":
    main()
