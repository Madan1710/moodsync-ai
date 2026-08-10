"""Plot helpers for the Streamlit UI.

Kept separate so we can unit-test or swap chart libs without touching app
logic. We use Plotly throughout because it's interactive in Streamlit and
shows up well in screen recordings of demos.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from moodsync.config import CANONICAL_EMOTIONS
from moodsync.utils.alignment import emojify


# --------------------------------------------------------------------------- #
#  Bar charts
# --------------------------------------------------------------------------- #
def emotion_bar_chart(
    distribution: np.ndarray,
    title: str,
    color: str = "#4C78A8",
    highlight_color: str = "#F58518",
) -> go.Figure:
    """Horizontal bar chart of a canonical 7-vector with the top emotion highlighted."""
    if distribution.shape[-1] != len(CANONICAL_EMOTIONS):
        raise ValueError("Expected canonical 7-vector")

    labels = [f"{emojify(e)} {e}" for e in CANONICAL_EMOTIONS]
    values = [float(v) * 100 for v in distribution]
    top_idx = int(np.argmax(distribution))
    colors = [highlight_color if i == top_idx else color for i in range(len(values))]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="Confidence (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=10, r=40, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def comparison_bar_chart(
    distributions: Dict[str, np.ndarray],
    title: str = "Per-modality emotion distributions",
) -> go.Figure:
    """Grouped horizontal bar chart comparing modalities."""
    fig = go.Figure()
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    labels = [f"{emojify(e)} {e}" for e in CANONICAL_EMOTIONS]

    for i, (name, dist) in enumerate(distributions.items()):
        fig.add_trace(
            go.Bar(
                name=name.capitalize(),
                y=labels,
                x=[float(v) * 100 for v in dist],
                orientation="h",
                marker_color=palette[i % len(palette)],
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis=dict(title="Confidence (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=10, r=20, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def valence_donut(valence: Dict[str, float]) -> go.Figure:
    """Compact donut chart showing positive/negative/neutral mass."""
    colors = {"positive": "#54A24B", "negative": "#E45756", "neutral": "#9D9D9D"}
    labels = list(valence.keys())
    values = [valence[k] * 100 for k in labels]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker_colors=[colors[k] for k in labels],
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="Valence breakdown",
        height=280,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --------------------------------------------------------------------------- #
#  Image overlays
# --------------------------------------------------------------------------- #
def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on top of the original face crop."""
    import cv2

    base = np.array(image.convert("RGB"))
    h, w = base.shape[:2]
    cam = cv2.resize(heatmap, (w, h))
    cam = (cam - cam.min()) / (cam.ptp() + 1e-9)
    cam_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * cam_color + (1 - alpha) * base).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def draw_face_box(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (245, 133, 24),
    thickness: int = 3,
) -> Image.Image:
    """Draw the face bounding box on the original image (for the UI)."""
    import cv2

    arr = np.array(image.convert("RGB")).copy()
    x, y, w, h = bbox
    cv2.rectangle(arr, (x, y), (x + w, y + h), color, thickness)
    return Image.fromarray(arr)


# --------------------------------------------------------------------------- #
#  Token attention HTML
# --------------------------------------------------------------------------- #
def render_token_attention_html(
    tokens: List[str],
    weights: List[float],
) -> str:
    """Render coloured-background tokens for the text-attention panel.

    Streamlit will display this with `st.markdown(html, unsafe_allow_html=True)`.
    """
    if not tokens:
        return "<i>No tokens to display.</i>"

    parts = ['<div style="line-height:2.0; font-size:1.05rem;">']
    for tok, w in zip(tokens, weights):
        # Map weight in [0,1] to an orange opacity background.
        opacity = max(0.05, min(1.0, float(w)))
        parts.append(
            f'<span style="background-color: rgba(245, 133, 24, {opacity:.2f}); '
            f'padding: 2px 5px; margin: 1px; border-radius: 4px;">'
            f"{tok}</span>"
        )
    parts.append("</div>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
#  Timeline (video)
# --------------------------------------------------------------------------- #
def timeline_chart(
    timestamps: List[float],
    distributions: List[np.ndarray],
    title: str = "Facial emotion over time",
) -> go.Figure:
    """Stacked-area chart of canonical emotion probs across video frames."""
    if len(distributions) == 0:
        return go.Figure()

    arr = np.stack(distributions, axis=0) * 100  # (T, 7)
    palette = [
        "#E45756", "#B279A2", "#9D755D", "#54A24B",
        "#9D9D9D", "#4C78A8", "#F58518",
    ]
    fig = go.Figure()
    for i, emo in enumerate(CANONICAL_EMOTIONS):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=arr[:, i],
                mode="lines",
                stackgroup="one",
                name=f"{emojify(emo)} {emo}",
                line=dict(width=0.3, color=palette[i]),
                fillcolor=palette[i],
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Confidence (%)",
        height=320,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


__all__ = [
    "emotion_bar_chart",
    "comparison_bar_chart",
    "valence_donut",
    "overlay_heatmap",
    "draw_face_box",
    "render_token_attention_html",
    "timeline_chart",
]
