"""Streamlit dashboard for the tremor detector.

Run with:
    UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python \
        uv run streamlit run ui/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tremor_detector as td

TRACKED_POINTS = (
    "index_tip",
    "middle_tip",
    "thumb_tip",
    "ring_tip",
    "pinky_tip",
    "hand_center",
)
DEFAULT_TRACKED_POINTS = ("index_tip", "middle_tip", "thumb_tip")
PRESETS = {
    "Balanced": {
        "min_signal_seconds": 12.0,
        "min_confidence": 0.55,
        "min_amplitude": 0.05,
        "agreement_count": 2,
    },
    "Strict": {
        "min_signal_seconds": 15.0,
        "min_confidence": 0.65,
        "min_amplitude": 0.08,
        "agreement_count": 2,
    },
    "Sensitive": {
        "min_signal_seconds": 8.0,
        "min_confidence": 0.45,
        "min_amplitude": 0.04,
        "agreement_count": 2,
    },
}


def init_state() -> None:
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("buffers", None)
    st.session_state.setdefault("last_metrics", {})
    st.session_state.setdefault("last_assessment", None)
    if not isinstance(st.session_state.last_metrics, dict):
        st.session_state.last_metrics = {}


def main() -> None:
    st.set_page_config(page_title="NeuroTrack Tremor Dashboard", layout="wide")
    init_state()

    st.title("NeuroTrack Tremor Dashboard")
    st.caption("Live webcam tremor analysis. Prototype only — not a medical diagnostic device.")

    with st.sidebar:
        st.subheader("Capture")
        camera_source = st.selectbox("Source", ("webcam", "oak"), index=0)
        camera_index = st.number_input("Camera index", min_value=0, max_value=8, value=0, step=1)

        points = st.multiselect(
            "Tracked points",
            TRACKED_POINTS,
            default=list(DEFAULT_TRACKED_POINTS),
        )
        if not points:
            st.warning("Select at least one tracked point.")
            points = [TRACKED_POINTS[0]]
        window_seconds = st.slider("Analysis window (s)", 5.0, 40.0, 20.0, 1.0)

        st.subheader("Classifier")
        preset_name = st.selectbox("Preset", tuple(PRESETS), index=0)

        st.divider()
        if st.button(
            "Stop" if st.session_state.running else "Start",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.running = not st.session_state.running
            st.rerun()

    feed_col, metrics_col = st.columns([3, 2])
    with feed_col:
        st.subheader("Live feed")
        st.empty()
    with metrics_col:
        st.subheader("Analysis")
        st.caption("Metrics will appear once capture starts.")


if __name__ == "__main__":
    main()