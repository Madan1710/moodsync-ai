"""Central configuration for MoodSyncAI.

All tunable values live here so we never sprinkle magic numbers in module code.
Change a model? Change one line. Change a threshold? Change one line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# --------------------------------------------------------------------------- #
# Canonical 7-emotion label space (FER2013 / Ekman + neutral)
# Vision and text models are picked specifically because they share this set.
# --------------------------------------------------------------------------- #
CANONICAL_EMOTIONS: Tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

# Aliases that may appear in different model checkpoints, mapped back to canonical.
EMOTION_ALIASES: dict[str, str] = {
    # vision (FER2013 ViT)
    "anger": "angry",
    # text (Ekman from j-hartmann/emotion-english-distilroberta-base)
    "joy": "happy",
    "sadness": "sad",
    # audio (HuBERT-superb-er has a 4-class subset)
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    "neu": "neutral",
}

# Valence groups for mismatch detection (positive vs negative vs neutral cues).
VALENCE_POSITIVE = {"happy", "surprise"}
VALENCE_NEGATIVE = {"angry", "disgust", "fear", "sad"}
VALENCE_NEUTRAL = {"neutral"}


@dataclass(frozen=True)
class ModelConfig:
    """Hugging Face model identifiers — kept small enough to run on CPU."""

    vision_model: str = "trpakov/vit-face-expression"
    text_model: str = "j-hartmann/emotion-english-distilroberta-base"
    audio_emotion_model: str = "superb/hubert-large-superb-er"
    asr_model: str = "tiny"  # whisper size: tiny|base|small|medium|large
    generator_model: str = "google/flan-t5-base"


@dataclass(frozen=True)
class FusionConfig:
    """Fusion + mismatch-detection thresholds."""

    # Heuristic fusion modality weights (sum to 1.0).
    weight_vision: float = 0.55
    weight_text: float = 0.45
    weight_audio: float = 0.0  # raised dynamically when audio is present

    # Mismatch trigger: KL divergence between distributions.
    kl_mismatch_threshold: float = 0.35

    # Or: top-1 emotions in opposing valence groups (always triggers).
    valence_mismatch_overrides_kl: bool = True

    # Confidence calibration — temperature scaling factor applied to logits.
    vision_temperature: float = 1.2
    text_temperature: float = 1.0


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def fusion_weights(self) -> Path:
        return self.project_root / "assets" / "fusion_mlp.pt"

    @property
    def synthetic_data(self) -> Path:
        return self.project_root / "assets" / "synthetic_fusion_data.pt"

    @property
    def cache_dir(self) -> Path:
        d = self.project_root / ".cache"
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass(frozen=True)
class UIConfig:
    """Streamlit UI tuning."""

    max_video_seconds: int = 15        # keep demos snappy
    timeline_fps_sample: int = 2       # process every Nth frame for video timeline
    chart_color_scheme: str = "viridis"


# Singleton instances — import these elsewhere.
MODEL_CONFIG = ModelConfig()
FUSION_CONFIG = FusionConfig()
PATH_CONFIG = PathConfig()
UI_CONFIG = UIConfig()


__all__ = [
    "CANONICAL_EMOTIONS",
    "EMOTION_ALIASES",
    "VALENCE_POSITIVE",
    "VALENCE_NEGATIVE",
    "VALENCE_NEUTRAL",
    "MODEL_CONFIG",
    "FUSION_CONFIG",
    "PATH_CONFIG",
    "UI_CONFIG",
    "ModelConfig",
    "FusionConfig",
    "PathConfig",
    "UIConfig",
]
