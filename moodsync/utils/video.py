"""Video helpers: sample frames at fixed FPS, split audio track.

Supports MP4/MOV/WEBM and exposes (frames, fps, duration) plus an optional
audio waveform when the file has an audio track.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VideoData:
    frames: List[Image.Image]
    timestamps: List[float]      # seconds, aligned with frames
    fps: float
    duration: float
    audio: Optional[np.ndarray]  # mono, float32, native sample rate
    audio_sr: Optional[int]


def load_video(
    path: str | Path,
    sample_fps: float = 2.0,
    max_seconds: Optional[float] = None,
    extract_audio: bool = True,
) -> VideoData:
    """Load a video and sample frames at `sample_fps`.

    Args:
        path: video file path
        sample_fps: how many frames per second to extract (lower = faster demo)
        max_seconds: cap on total video duration to process
        extract_audio: whether to pull the audio track too

    Returns:
        VideoData with sampled frames, timestamps, optional audio.
    """
    import cv2

    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0.0

    if max_seconds is not None:
        duration = min(duration, max_seconds)

    sample_stride = max(1, int(round(src_fps / sample_fps)))

    frames: List[Image.Image] = []
    timestamps: List[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / src_fps
        if max_seconds is not None and ts > max_seconds:
            break
        if idx % sample_stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            timestamps.append(round(ts, 2))
        idx += 1
    cap.release()

    audio, audio_sr = (None, None)
    if extract_audio:
        audio, audio_sr = _extract_audio(path, max_seconds=max_seconds)

    return VideoData(
        frames=frames,
        timestamps=timestamps,
        fps=src_fps,
        duration=duration,
        audio=audio,
        audio_sr=audio_sr,
    )


def _extract_audio(
    path: str,
    max_seconds: Optional[float],
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Pull mono audio waveform via librosa (uses ffmpeg under the hood)."""
    try:
        import librosa

        audio, sr = librosa.load(path, sr=None, mono=True, duration=max_seconds)
        if audio.size == 0:
            return None, None
        return audio.astype(np.float32), int(sr)
    except Exception as e:
        logger.warning("Audio extraction failed (no audio track?): %s", e)
        return None, None


def write_uploaded_to_tempfile(uploaded_file, suffix: str = ".mp4") -> Path:
    """Streamlit gives us an UploadedFile-like; write it somewhere we can open."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


__all__ = ["VideoData", "load_video", "write_uploaded_to_tempfile"]
