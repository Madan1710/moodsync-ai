"""Reusable Streamlit UI components.

Keeps the main app file focused on flow rather than markup.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from moodsync.models.fusion import FusionResult
from moodsync.models.text import TextResult
from moodsync.models.vision import VisionResult
from moodsync.utils.alignment import emojify
from moodsync.utils.visualization import (
    comparison_bar_chart,
    emotion_bar_chart,
    overlay_heatmap,
    render_token_attention_html,
    valence_donut,
)


# --------------------------------------------------------------------------- #
#  Top-level result panel (used after every analysis)
# --------------------------------------------------------------------------- #
def render_fusion_header(fusion: FusionResult) -> None:
    """The big summary strip at the top: fusion verdict + mismatch badge."""
    col1, col2 = st.columns([3, 1])
    with col1:
        emo = fusion.top_label
        st.markdown(
            f"### {emojify(emo)} Fusion verdict: **{emo.upper()}** "
            f"<span style='color:#888;font-size:0.95rem'>"
            f"({fusion.top_confidence:.0%} confidence · {fusion.strategy} fusion)</span>",
            unsafe_allow_html=True,
        )
    with col2:
        if fusion.mismatch:
            st.markdown(
                "<div style='background:#F58518;color:white;padding:10px 14px;"
                "border-radius:8px;font-weight:600;text-align:center;'>"
                "⚠️ MISMATCH DETECTED</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#54A24B;color:white;padding:10px 14px;"
                "border-radius:8px;font-weight:600;text-align:center;'>"
                "✓ MODALITIES ALIGNED</div>",
                unsafe_allow_html=True,
            )


def render_modality_comparison(fusion: FusionResult) -> None:
    """Per-modality grouped bar chart + valence donut + raw mismatch reason."""
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(
            comparison_bar_chart(fusion.modality_distributions),
            use_container_width=True,
            key=f"comp_{id(fusion)}",
        )
    with c2:
        st.plotly_chart(
            valence_donut(fusion.valence),
            use_container_width=True,
            key=f"val_{id(fusion)}",
        )

    if fusion.mismatch:
        st.warning(f"**Why mismatch?** {fusion.mismatch_reason}")
    else:
        st.info(f"**Cross-modality check:** {fusion.mismatch_reason}")


def render_summary(summary_text: str) -> None:
    """Generative summary, formatted as a quote-style block."""
    st.markdown("#### 📝 Generative summary")
    st.markdown(
        f"<div style='background:#F5F5F5;border-left:4px solid #4C78A8;"
        f"padding:14px 18px;border-radius:6px;font-size:1.05rem;line-height:1.5;'>"
        f"{summary_text}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Per-modality detail panels
# --------------------------------------------------------------------------- #
def render_vision_detail(result: VisionResult) -> None:
    st.markdown("#### 👁️ Visual emotion")
    cols = st.columns([1, 1])

    with cols[0]:
        # Show the face crop (or full image if no detection).
        st.image(
            result.face.image,
            caption=(
                f"Detected face (confidence {result.face.confidence:.2f})"
                if result.face.detected
                else "No face detected — using whole frame"
            ),
            use_container_width=True,
        )
        if result.heatmap is not None:
            st.image(
                overlay_heatmap(result.face.image, result.heatmap),
                caption="Grad-CAM: regions the model attended to",
                use_container_width=True,
            )

    with cols[1]:
        st.plotly_chart(
            emotion_bar_chart(result.distribution, title="Visual emotion distribution"),
            use_container_width=True,
            key=f"vis_{id(result)}",
        )


def render_text_detail(result: TextResult) -> None:
    st.markdown("#### 💬 Text emotion")
    cols = st.columns([1, 1])

    with cols[0]:
        st.markdown(f"**Input:** _{result.cleaned_text or '(empty)'}_")
        if result.tokens:
            st.markdown("**Token attention** _(brighter = more salient)_")
            st.markdown(
                render_token_attention_html(result.tokens, result.attention),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No tokens to display.")

    with cols[1]:
        st.plotly_chart(
            emotion_bar_chart(result.distribution, title="Textual emotion distribution"),
            use_container_width=True,
            key=f"txt_{id(result)}",
        )


def render_audio_detail(audio_result, audio_bytes: Optional[bytes] = None) -> None:
    st.markdown("#### 🎤 Audio emotion")
    cols = st.columns([1, 1])
    with cols[0]:
        if audio_bytes is not None:
            st.audio(audio_bytes)
        st.markdown(f"**Whisper transcript:** _{audio_result.transcript or '(empty)'}_")
        st.caption(f"Duration: {audio_result.duration_sec:.1f}s")
    with cols[1]:
        st.plotly_chart(
            emotion_bar_chart(audio_result.distribution, title="Acoustic emotion distribution"),
            use_container_width=True,
            key=f"aud_{id(audio_result)}",
        )


__all__ = [
    "render_fusion_header",
    "render_modality_comparison",
    "render_summary",
    "render_vision_detail",
    "render_text_detail",
    "render_audio_detail",
]
