"""Audio modality: Whisper transcription + speech emotion recognition.

Pipeline:
    audio waveform
        ├── Whisper (tiny) -> transcript text  -> feeds the text model too
        └── HuBERT-superb-er -> 4-class emotion -> projected to canonical 7

Audio is graded as an *extended feature* (extra marks). Combining it gives
us 3 modalities for the fusion layer.

Limitation noted for the report: HuBERT-superb-er has only 4 classes
{angry, happy, neutral, sad}. We project these into the 7-class canonical
space; the missing classes (disgust, fear, surprise) get zero mass — which
is fine for averaging, since the other modalities can supply that mass.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from moodsync.config import MODEL_CONFIG
from moodsync.utils.alignment import to_canonical_distribution

logger = logging.getLogger(__name__)

TARGET_SR = 16_000


@dataclass
class AudioResult:
    """Output of audio pipeline."""

    distribution: np.ndarray            # canonical 7-vector (from acoustic emotion)
    raw_labels: List[str]
    raw_probs: List[float]
    transcript: str                     # Whisper transcript
    duration_sec: float


class AudioEmotionModel:
    """Lazy-loaded HuBERT speech-emotion + Whisper ASR."""

    def __init__(self) -> None:
        self._emotion_model = None
        self._emotion_extractor = None
        self._whisper = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_emotion(self) -> None:
        if self._emotion_model is not None:
            return
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        logger.info("Loading audio emotion model: %s", MODEL_CONFIG.audio_emotion_model)
        self._emotion_extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_CONFIG.audio_emotion_model
        )
        self._emotion_model = AutoModelForAudioClassification.from_pretrained(
            MODEL_CONFIG.audio_emotion_model
        ).to(self._device)
        self._emotion_model.eval()

    def _ensure_whisper(self) -> None:
        if self._whisper is not None:
            return
        import whisper

        logger.info("Loading Whisper: %s", MODEL_CONFIG.asr_model)
        self._whisper = whisper.load_model(MODEL_CONFIG.asr_model, device=self._device)

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcribe: bool = True,
    ) -> AudioResult:
        """Predict emotion + transcribe an audio waveform."""
        wav = _ensure_mono_16k(audio, sample_rate)
        duration = len(wav) / TARGET_SR

        # ---- emotion ----
        self._ensure_emotion()
        inputs = self._emotion_extractor(
            wav, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        ).to(self._device)
        with torch.no_grad():
            logits = self._emotion_model(**inputs).logits[0].cpu().numpy()
        probs = _softmax(logits)

        id2label = self._emotion_model.config.id2label
        labels = [id2label[i] for i in range(len(probs))]
        canonical = to_canonical_distribution(labels, probs.tolist())

        # ---- transcript ----
        transcript = ""
        if transcribe:
            self._ensure_whisper()
            try:
                # Whisper expects float32 mono 16k.
                result = self._whisper.transcribe(
                    wav.astype(np.float32),
                    fp16=False,
                    language=None,
                )
                transcript = result.get("text", "").strip()
            except Exception as e:  # pragma: no cover
                logger.warning("Whisper failed: %s", e)
                transcript = ""

        return AudioResult(
            distribution=canonical,
            raw_labels=labels,
            raw_probs=probs.tolist(),
            transcript=transcript,
            duration_sec=duration,
        )

    def transcribe_only(self, audio: np.ndarray, sample_rate: int) -> str:
        wav = _ensure_mono_16k(audio, sample_rate)
        self._ensure_whisper()
        try:
            result = self._whisper.transcribe(wav.astype(np.float32), fp16=False)
            return result.get("text", "").strip()
        except Exception as e:  # pragma: no cover
            logger.warning("Whisper failed: %s", e)
            return ""


# --------------------------------------------------------------------------- #
#  Audio helpers
# --------------------------------------------------------------------------- #
def _ensure_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Coerce to mono float32 at 16 kHz, the universal speech-model input."""
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1) if audio.shape[-1] < audio.shape[0] else audio.mean(axis=0)
    if sample_rate != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SR)
    # Peak-normalise to avoid feature-extractor edge cases.
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak
    return audio


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# Module-level singleton.
_MODEL: Optional[AudioEmotionModel] = None


def get_audio_model() -> AudioEmotionModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = AudioEmotionModel()
    return _MODEL


__all__ = ["AudioEmotionModel", "AudioResult", "get_audio_model"]
