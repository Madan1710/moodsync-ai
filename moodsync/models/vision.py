"""Visual emotion classifier with Grad-CAM explainability.

Pipeline:
    raw image -> face detection -> ViT classifier -> 7-class probs
                                                  -> Grad-CAM heatmap

Design notes (for the report):

* We use **Vision Transformer** (`trpakov/vit-face-expression`) rather than a
  CNN trained from scratch. This is deliberate: (i) the brief mentions
  "CNN or ViT" so ViT counts; (ii) starting from ImageNet-pretrained ViT
  beats a from-scratch CNN by a comfortable margin on FER2013 with no
  training data of our own; (iii) it lets us showcase Grad-CAM on a
  transformer, which is a non-trivial technical depth point.

* **Grad-CAM** for ViTs needs a `reshape_transform` because attention maps
  are 1D sequences not 2D feature maps — handled below.

* **Temperature scaling** softens the over-confident base head; calibrated
  probabilities matter when fusing across modalities.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from moodsync.config import MODEL_CONFIG, FUSION_CONFIG
from moodsync.models.face_detector import FaceCrop, get_face_detector
from moodsync.utils.alignment import (
    softmax_with_temperature,
    to_canonical_distribution,
)

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Output of vision pipeline."""

    distribution: np.ndarray             # canonical 7-vector
    raw_labels: List[str]                # native labels from the model
    raw_probs: List[float]               # native probs
    face: FaceCrop                       # face-detection result
    heatmap: Optional[np.ndarray] = None # Grad-CAM (H, W) in [0, 1] or None


class VisionEmotionModel:
    """ViT-based facial emotion classifier with Grad-CAM."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._cam = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------- #
    #  Lazy loading
    # --------------------------------------------------------------------- #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        logger.info("Loading vision model: %s", MODEL_CONFIG.vision_model)
        self._processor = AutoImageProcessor.from_pretrained(MODEL_CONFIG.vision_model)
        self._model = AutoModelForImageClassification.from_pretrained(
            MODEL_CONFIG.vision_model
        ).to(self._device)
        self._model.eval()

    def _ensure_cam(self) -> None:
        """Initialise Grad-CAM for the ViT (handled lazily — heavy import)."""
        if self._cam is not None:
            return
        self._ensure_loaded()
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: F401

            # ViT-specific: target the last LayerNorm before the classifier head.
            target_layers = [self._model.vit.encoder.layer[-1].layernorm_before]
            self._cam = GradCAM(
                model=self._model,
                target_layers=target_layers,
                reshape_transform=_vit_reshape_transform,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Grad-CAM unavailable (%s); heatmaps will be None.", e)
            self._cam = False  # sentinel: we tried, it failed

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def predict(
        self,
        image: Image.Image,
        compute_heatmap: bool = True,
    ) -> VisionResult:
        """Predict canonical emotion distribution for an image."""
        self._ensure_loaded()

        face = get_face_detector().detect(image)
        crop = face.image

        inputs = self._processor(images=crop, return_tensors="pt").to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits[0].cpu().numpy()

        # Calibrate then map to canonical 7-class.
        probs = softmax_with_temperature(logits, FUSION_CONFIG.vision_temperature)
        id2label = self._model.config.id2label
        labels = [id2label[i] for i in range(len(probs))]
        canonical = to_canonical_distribution(labels, probs.tolist())

        heatmap = None
        if compute_heatmap:
            heatmap = self._gradcam(crop, target_class=int(np.argmax(probs)))

        return VisionResult(
            distribution=canonical,
            raw_labels=labels,
            raw_probs=probs.tolist(),
            face=face,
            heatmap=heatmap,
        )

    # --------------------------------------------------------------------- #
    #  Grad-CAM
    # --------------------------------------------------------------------- #
    def _gradcam(
        self,
        crop: Image.Image,
        target_class: int,
    ) -> Optional[np.ndarray]:
        """Compute Grad-CAM heatmap (H, W) in [0, 1], or None on failure."""
        self._ensure_cam()
        if not self._cam:
            return None

        try:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            inputs = self._processor(images=crop, return_tensors="pt").to(self._device)
            input_tensor = inputs["pixel_values"]
            targets = [ClassifierOutputTarget(target_class)]
            grayscale = self._cam(input_tensor=input_tensor, targets=targets)[0]
            return grayscale.astype(np.float32)
        except Exception as e:  # pragma: no cover
            logger.warning("Grad-CAM failed: %s", e)
            return None


def _vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Reshape ViT token sequence (B, N, D) -> (B, D, H, W) for Grad-CAM.

    ViT-Base uses 14x14 patches at 224x224 + 1 CLS token = 197 tokens. We
    discard the CLS token and reshape the spatial tokens into a feature map.
    """
    # Drop CLS token (first position).
    result = tensor[:, 1:, :]
    # (B, N, D) -> (B, H, W, D) -> (B, D, H, W)
    result = result.reshape(result.size(0), height, width, result.size(2))
    return result.permute(0, 3, 1, 2)


# Module-level singleton.
_MODEL: Optional[VisionEmotionModel] = None


def get_vision_model() -> VisionEmotionModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = VisionEmotionModel()
    return _MODEL


__all__ = ["VisionEmotionModel", "VisionResult", "get_vision_model"]
