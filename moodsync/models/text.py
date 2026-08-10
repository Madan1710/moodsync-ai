"""Text emotion classifier with token-attention visualisation.

Pipeline:
    text -> tokenize -> DistilRoBERTa-emotion -> 7-class probs
                                              -> per-token attention scores

Why this model: `j-hartmann/emotion-english-distilroberta-base` outputs the
exact 7 Ekman emotions {anger, disgust, fear, joy, neutral, sadness, surprise},
which align 1:1 with our vision model's FER2013 labels. That alignment makes
fusion *meaningful* rather than ad-hoc — it's the single most important
architectural decision in the project.

Attention visualisation uses the last layer's [CLS]-row attention averaged
across heads. There are fancier methods (rollout, gradient×attention) but
[CLS]-row from the last layer is the standard, defensible choice you can
explain in 30 seconds during the viva.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from moodsync.config import MODEL_CONFIG, FUSION_CONFIG
from moodsync.utils.alignment import (
    softmax_with_temperature,
    to_canonical_distribution,
)

logger = logging.getLogger(__name__)


@dataclass
class TextResult:
    """Output of text pipeline."""

    distribution: np.ndarray            # canonical 7-vector
    raw_labels: List[str]               # native labels
    raw_probs: List[float]              # native probs
    tokens: List[str]                   # display-friendly tokens
    attention: List[float]              # per-token salience in [0, 1], same len as tokens
    cleaned_text: str                   # text used for inference (after stripping)


class TextEmotionModel:
    """DistilRoBERTa emotion model with last-layer [CLS] attention extraction."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading text model: %s", MODEL_CONFIG.text_model)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.text_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CONFIG.text_model,
            output_attentions=True,
        ).to(self._device)
        self._model.eval()

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def predict(self, text: str) -> TextResult:
        """Predict canonical emotion distribution + token-level salience."""
        self._ensure_loaded()
        cleaned = (text or "").strip()
        if not cleaned:
            uniform = np.full(7, 1.0 / 7)
            return TextResult(
                distribution=uniform,
                raw_labels=list(self._model.config.id2label.values()),
                raw_probs=uniform.tolist(),
                tokens=[],
                attention=[],
                cleaned_text=cleaned,
            )

        inputs = self._tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            return_attention_mask=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits[0].cpu().numpy()
        probs = softmax_with_temperature(logits, FUSION_CONFIG.text_temperature)

        id2label = self._model.config.id2label
        labels = [id2label[i] for i in range(len(probs))]
        canonical = to_canonical_distribution(labels, probs.tolist())

        tokens, attention = self._extract_attention(inputs, outputs)

        return TextResult(
            distribution=canonical,
            raw_labels=labels,
            raw_probs=probs.tolist(),
            tokens=tokens,
            attention=attention,
            cleaned_text=cleaned,
        )

    # --------------------------------------------------------------------- #
    #  Attention extraction
    # --------------------------------------------------------------------- #
    def _extract_attention(
        self,
        inputs: dict,
        outputs,
    ) -> Tuple[List[str], List[float]]:
        """Extract human-readable per-token attention weights.

        Returns:
            (tokens, attention) — special tokens stripped, weights min-max
            scaled to [0, 1] across the displayed tokens.
        """
        if not hasattr(outputs, "attentions") or outputs.attentions is None:
            return [], []

        # Last layer, mean over heads, [CLS]-row attention to other tokens.
        attn = outputs.attentions[-1][0]              # (heads, seq, seq)
        cls_attn = attn.mean(dim=0)[0].cpu().numpy()  # (seq,)

        token_ids = inputs["input_ids"][0].cpu().numpy()
        all_tokens = self._tokenizer.convert_ids_to_tokens(token_ids)

        # Filter out special tokens; keep order.
        special = set(self._tokenizer.all_special_tokens)
        tokens, weights = [], []
        for tok, w in zip(all_tokens, cls_attn):
            if tok in special:
                continue
            # DistilRoBERTa uses BPE with leading 'Ġ' for word-starts; clean it.
            display = tok.replace("Ġ", " ").lstrip()
            if not display:
                continue
            tokens.append(display)
            weights.append(float(w))

        if not tokens:
            return [], []

        # Min-max scale for display.
        w = np.asarray(weights, dtype=np.float64)
        w_range = w.max() - w.min()
        if w_range > 1e-9:
            w = (w - w.min()) / w_range
        else:
            w = np.zeros_like(w)
        return tokens, w.tolist()


# Module-level singleton.
_MODEL: Optional[TextEmotionModel] = None


def get_text_model() -> TextEmotionModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = TextEmotionModel()
    return _MODEL


__all__ = ["TextEmotionModel", "TextResult", "get_text_model"]
