"""MoodSyncAI — Streamlit demo entry point.

Run:
    streamlit run app.py

Tabs:
  📸 Image + Text  — primary flow from the assignment brief
  🎥 Video         — frame-level timeline + audio fusion (extended)
  🎤 Audio + Text  — third modality with Whisper transcription (extended)
  📷 Webcam        — real-time short clip capture (extended)
  ℹ️ About         — architecture and design notes
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

from moodsync.config import UI_CONFIG
from moodsync.models.fusion import get_learned_fusion, heuristic_fuse
from moodsync.models.generator import get_generator
from moodsync.ui.components import (
    render_audio_detail,
    render_fusion_header,
    render_modality_comparison,
    render_summary,
    render_text_detail,
    render_vision_detail,
)
from moodsync.utils.visualization import timeline_chart

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(
    page_title="MoodSyncAI · Multi-modal Emotion Analyser",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
#  Cached model loaders — Streamlit re-runs the script on every interaction;
#  these decorators ensure we don't reload models from disk every time.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading vision model (ViT face emotion)…")
def _vision_model():
    from moodsync.models.vision import get_vision_model
    return get_vision_model()


@st.cache_resource(show_spinner="Loading text model (DistilRoBERTa emotion)…")
def _text_model():
    from moodsync.models.text import get_text_model
    return get_text_model()


@st.cache_resource(show_spinner="Loading audio model (HuBERT + Whisper)…")
def _audio_model():
    from moodsync.models.audio import get_audio_model
    return get_audio_model()


@st.cache_resource(show_spinner="Loading generator (flan-T5)…")
def _generator():
    return get_generator()


@st.cache_resource
def _learned_fusion():
    return get_learned_fusion()


# --------------------------------------------------------------------------- #
#  Sidebar — settings shared across tabs
# --------------------------------------------------------------------------- #
def sidebar() -> dict:
    st.sidebar.title("⚙️ MoodSyncAI")
    st.sidebar.caption(
        "Multi-modal sentiment and emotion analyser. "
        "Vision (ViT) + Text (RoBERTa) + Audio (Whisper/HuBERT) → fusion → summary."
    )
    st.sidebar.divider()

    fusion_choice = st.sidebar.radio(
        "Fusion strategy",
        options=["Heuristic (weighted average)", "Learned (small MLP)"],
        index=0,
        help=(
            "Heuristic fusion always works. Learned fusion uses a trained "
            "MLP — falls back to heuristic if weights file is missing."
        ),
    )

    use_llm = st.sidebar.toggle(
        "Use flan-T5 polish for summary",
        value=True,
        help="Off = template-only summary (faster, fully deterministic).",
    )
    show_explain = st.sidebar.toggle(
        "Show explainability",
        value=True,
        help="Grad-CAM heatmap for vision; attention weights for text.",
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Tip:** the most interesting demos are *mismatched* — a smiling "
        "face + sad text, or a sad face + the assignment-brief sentence "
        "_'No, I think the project is going really well.'_"
    )
    return {
        "fusion_choice": fusion_choice,
        "use_llm": use_llm,
        "show_explain": show_explain,
    }


# --------------------------------------------------------------------------- #
#  Helper: run fusion with the chosen strategy
# --------------------------------------------------------------------------- #
def _do_fusion(settings, vision=None, text=None, audio=None):
    if settings["fusion_choice"].startswith("Learned"):
        fusion = _learned_fusion().fuse(vision=vision, text=text, audio=audio)
    else:
        fusion = heuristic_fuse(vision=vision, text=text, audio=audio)
    return fusion


# --------------------------------------------------------------------------- #
#  TAB 1 — Image + Text  (assignment-brief flow)
# --------------------------------------------------------------------------- #
def tab_image_text(settings: dict) -> None:
    st.subheader("📸 Image + Text")
    st.caption(
        "The flow described in the assignment brief: upload a face photo, "
        "type what the person said, see whether vision and text agree."
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        uploaded = st.file_uploader(
            "Upload a face photo (jpg, png, webp)",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_upload",
        )
    with col_right:
        text = st.text_area(
            "What did they say?",
            value="No, I think the project is going really well.",
            help="The exact sample sentence from the assignment brief is pre-filled.",
            key="img_text",
        )

    if not st.button("🚀 Analyse", type="primary", key="img_btn"):
        return

    if uploaded is None:
        st.error("Please upload a face photo first.")
        return
    if not text.strip():
        st.error("Please type what the person said.")
        return

    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    with st.spinner("Running vision and text models…"):
        v_result = _vision_model().predict(image, compute_heatmap=settings["show_explain"])
        t_result = _text_model().predict(text)

    fusion = _do_fusion(settings, vision=v_result.distribution, text=t_result.distribution)

    with st.spinner("Generating summary…"):
        gen = _generator() if settings["use_llm"] else _NO_LLM_GENERATOR
        summary = gen.generate(fusion, spoken_text=text)

    st.divider()
    render_fusion_header(fusion)
    render_summary(summary)
    st.divider()
    render_modality_comparison(fusion)
    st.divider()
    if settings["show_explain"]:
        render_vision_detail(v_result)
        render_text_detail(t_result)


# --------------------------------------------------------------------------- #
#  TAB 2 — Video (frame timeline + audio fusion)
# --------------------------------------------------------------------------- #
def tab_video(settings: dict) -> None:
    st.subheader("🎥 Video — emotion over time + audio fusion")
    st.caption(
        f"Upload a short clip (≤ {UI_CONFIG.max_video_seconds}s). "
        "Frames are sampled to build an emotion timeline; the audio track "
        "is transcribed by Whisper and fused with the average facial emotion."
    )

    uploaded = st.file_uploader(
        "Upload a short video (mp4, webm, mov)",
        type=["mp4", "webm", "mov"],
        key="vid_upload",
    )
    if uploaded is None or not st.button("🚀 Analyse video", type="primary", key="vid_btn"):
        return

    from moodsync.utils.video import load_video, write_uploaded_to_tempfile

    suffix = "." + uploaded.name.rsplit(".", 1)[-1]
    tmp = write_uploaded_to_tempfile(uploaded, suffix=suffix)

    with st.spinner("Decoding frames + audio…"):
        video = load_video(
            tmp,
            sample_fps=UI_CONFIG.timeline_fps_sample,
            max_seconds=UI_CONFIG.max_video_seconds,
        )

    if not video.frames:
        st.error("No frames decoded — is the video file valid?")
        return

    st.video(str(tmp))

    # ---- per-frame vision emotion ----
    with st.spinner(f"Running vision on {len(video.frames)} frames…"):
        per_frame = [
            _vision_model().predict(f, compute_heatmap=False).distribution
            for f in video.frames
        ]
    avg_vision = np.mean(np.stack(per_frame, axis=0), axis=0)

    # ---- audio (transcript + emotion) ----
    audio_result = None
    transcript = ""
    if video.audio is not None and video.audio_sr:
        with st.spinner("Transcribing + audio emotion…"):
            audio_result = _audio_model().predict(video.audio, video.audio_sr)
            transcript = audio_result.transcript

    # ---- text from transcript ----
    t_result = None
    if transcript:
        with st.spinner("Running text emotion on transcript…"):
            t_result = _text_model().predict(transcript)

    # ---- fusion ----
    fusion = _do_fusion(
        settings,
        vision=avg_vision,
        text=t_result.distribution if t_result else None,
        audio=audio_result.distribution if audio_result else None,
    )

    with st.spinner("Generating summary…"):
        gen = _generator() if settings["use_llm"] else _NO_LLM_GENERATOR
        summary = gen.generate(fusion, transcript=transcript)

    st.divider()
    render_fusion_header(fusion)
    render_summary(summary)
    st.divider()

    st.markdown("#### ⏱ Emotion timeline (per frame)")
    st.plotly_chart(
        timeline_chart(video.timestamps, per_frame, title="Facial emotion across frames"),
        use_container_width=True,
    )

    render_modality_comparison(fusion)
    st.divider()

    if audio_result:
        render_audio_detail(audio_result)
    if t_result:
        render_text_detail(t_result)


# --------------------------------------------------------------------------- #
#  TAB 3 — Audio + Text (Whisper transcription as the third modality)
# --------------------------------------------------------------------------- #
def tab_audio(settings: dict) -> None:
    st.subheader("🎤 Audio + Text")
    st.caption(
        "Upload an audio clip — Whisper transcribes it (feeding the text "
        "channel) and HuBERT classifies the speech emotion directly. "
        "Optionally upload a face photo to add the vision modality too."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        audio_file = st.file_uploader(
            "Audio clip (wav, mp3, m4a)",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="aud_upload",
        )
    with col_b:
        face_file = st.file_uploader(
            "Optional face photo for tri-modal fusion",
            type=["jpg", "jpeg", "png", "webp"],
            key="aud_face",
        )

    if audio_file is None or not st.button("🚀 Analyse audio", type="primary", key="aud_btn"):
        return

    import librosa

    audio_bytes = audio_file.read()
    waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

    with st.spinner("Whisper transcription + audio emotion…"):
        a_result = _audio_model().predict(waveform, sr)

    if not a_result.transcript:
        st.warning("Whisper produced no transcript. Continuing with audio + face only.")

    t_result = None
    if a_result.transcript:
        with st.spinner("Text emotion on transcript…"):
            t_result = _text_model().predict(a_result.transcript)

    v_result = None
    if face_file is not None:
        image = Image.open(io.BytesIO(face_file.read())).convert("RGB")
        with st.spinner("Vision emotion…"):
            v_result = _vision_model().predict(image, compute_heatmap=settings["show_explain"])

    fusion = _do_fusion(
        settings,
        vision=v_result.distribution if v_result else None,
        text=t_result.distribution if t_result else None,
        audio=a_result.distribution,
    )

    with st.spinner("Generating summary…"):
        gen = _generator() if settings["use_llm"] else _NO_LLM_GENERATOR
        summary = gen.generate(fusion, transcript=a_result.transcript)

    st.divider()
    render_fusion_header(fusion)
    render_summary(summary)
    st.divider()
    render_modality_comparison(fusion)
    st.divider()
    render_audio_detail(a_result, audio_bytes=audio_bytes)
    if t_result:
        render_text_detail(t_result)
    if v_result:
        render_vision_detail(v_result)


# --------------------------------------------------------------------------- #
#  TAB 4 — Webcam (single-frame snap)
# --------------------------------------------------------------------------- #
def tab_webcam(settings: dict) -> None:
    st.subheader("📷 Webcam — quick snap")
    st.caption(
        "Take a snapshot from your webcam and pair it with text. "
        "Streamlit's built-in camera input keeps the deployment story simple "
        "and works on Streamlit Cloud."
    )

    snap = st.camera_input("Take a photo")
    text = st.text_area(
        "What are you saying?",
        value="I'm absolutely delighted with how this turned out!",
        key="webcam_text",
    )
    if snap is None or not st.button("🚀 Analyse snapshot", type="primary", key="cam_btn"):
        return

    image = Image.open(snap).convert("RGB")

    with st.spinner("Vision + text…"):
        v_result = _vision_model().predict(image, compute_heatmap=settings["show_explain"])
        t_result = _text_model().predict(text) if text.strip() else None

    fusion = _do_fusion(
        settings,
        vision=v_result.distribution,
        text=t_result.distribution if t_result else None,
    )

    with st.spinner("Generating summary…"):
        gen = _generator() if settings["use_llm"] else _NO_LLM_GENERATOR
        summary = gen.generate(fusion, spoken_text=text)

    st.divider()
    render_fusion_header(fusion)
    render_summary(summary)
    st.divider()
    render_modality_comparison(fusion)
    st.divider()
    render_vision_detail(v_result)
    if t_result:
        render_text_detail(t_result)


# --------------------------------------------------------------------------- #
#  TAB 5 — About / architecture
# --------------------------------------------------------------------------- #
def tab_about() -> None:
    st.subheader("ℹ️ About MoodSyncAI")

    st.markdown(
        """
**MoodSyncAI** is a multi-modal emotion analyser implementing the
Data-Analytics-3 final project brief.

#### Components
| Layer | Model | Role |
|-------|-------|------|
| Vision | `trpakov/vit-face-expression` (ViT) | Facial emotion (7-class FER2013) |
| Text | `j-hartmann/emotion-english-distilroberta-base` | Textual emotion (7-class Ekman) |
| Audio (optional) | `superb/hubert-large-superb-er` | Speech emotion |
| ASR (optional) | `openai/whisper-tiny` | Audio → transcript |
| Fusion | Heuristic + Learned MLP | Cross-modal combination |
| Generator | `google/flan-t5-base` | Natural-language summary |

#### Why these choices
* **Label alignment.** Vision and text models share an identical 7-emotion
  vocabulary, so fusion is principled (not string-matching).
* **Calibration.** Pretrained classifier heads tend to over-estimate
  confidence; we apply temperature scaling so KL-divergence between
  modalities is meaningful.
* **Mismatch detection.** Two complementary signals — Jensen–Shannon
  divergence between modality distributions, and a hard valence-conflict
  rule that overrides JS when one modality reads positive and another
  negative.
* **Explainability.** Grad-CAM for the ViT (with reshape transform for
  attention-token-to-spatial-map conversion) and last-layer [CLS]-row
  attention for the text model.
* **Reliability.** The generator falls back to a deterministic template
  if the LLM is unavailable, so the demo never silently fails.
        """
    )

    st.markdown("#### Pipeline diagram")
    try:
        with open("docs/architecture_diagram.svg", "r") as f:
            st.image(f.read(), use_container_width=True)
    except Exception:
        st.caption("(architecture diagram will appear after `docs/architecture_diagram.svg` is generated)")


# --------------------------------------------------------------------------- #
#  Sentinel for "use template only" mode
# --------------------------------------------------------------------------- #
class _NoLLMGenerator:
    def generate(self, fusion, spoken_text=None, transcript=None):
        from moodsync.models.generator import SummaryGenerator
        sg = SummaryGenerator(use_llm=False)
        return sg.generate(fusion, spoken_text=spoken_text, transcript=transcript)


_NO_LLM_GENERATOR = _NoLLMGenerator()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🎭 MoodSyncAI")
    st.caption(
        "Multi-modal sentiment & emotion analyser · Vision + Text + Audio fusion"
    )

    settings = sidebar()

    tabs = st.tabs(["📸 Image + Text", "🎥 Video", "🎤 Audio", "📷 Webcam", "ℹ️ About"])
    with tabs[0]:
        tab_image_text(settings)
    with tabs[1]:
        tab_video(settings)
    with tabs[2]:
        tab_audio(settings)
    with tabs[3]:
        tab_webcam(settings)
    with tabs[4]:
        tab_about()


if __name__ == "__main__":
    main()
