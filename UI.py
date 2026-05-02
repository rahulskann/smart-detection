"""Streamlit dashboard for the tremor detector.

Run with:
    UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python \
        uv run streamlit run ui/dashboard.py
"""

from __future__ import annotations

import sys
import time
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
SETTLING_SECONDS = 2.0
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
    st.session_state.setdefault("hand_present_since", None)
    st.session_state.setdefault("was_hand_present", False)
    if not isinstance(st.session_state.last_metrics, dict):
        st.session_state.last_metrics = {}
    if st.session_state.buffers is not None and not isinstance(st.session_state.buffers, dict):
        st.session_state.buffers = None


def reset_buffers(window_seconds: float, points: tuple[str, ...]) -> None:
    st.session_state.buffers = {
        point: td.MotionBuffer(window_seconds)
        for point in points
    }
    st.session_state.last_metrics = {}
    st.session_state.last_assessment = None
    st.session_state.hand_present_since = None
    st.session_state.was_hand_present = False


def ensure_buffers(window_seconds: float, points: tuple[str, ...]) -> dict[str, td.MotionBuffer]:
    if st.session_state.buffers is None:
        reset_buffers(window_seconds, points)
    buffers: dict[str, td.MotionBuffer] = st.session_state.buffers
    for point in points:
        buffers.setdefault(point, td.MotionBuffer(window_seconds))
        buffers[point].window_seconds = window_seconds
    for point in list(buffers):
        if point not in points:
            del buffers[point]
    return buffers


def render_waveform_image(waveform: np.ndarray, width: int = 560, height: int = 180) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    if td.cv2 is not None:
        td.draw_waveform(canvas, waveform, (0, 0), (width, height))
        return td.cv2.cvtColor(canvas, td.cv2.COLOR_BGR2RGB)
    return canvas


def format_frequency(frequency: float | None) -> str:
    return "—" if frequency is None else f"{frequency:0.2f} Hz"


def render_metrics(slot, metrics: td.TremorMetrics, point_name: str) -> None:
    with slot.container():
        cols = st.columns(2)
        cols[0].metric("Dominant motion", format_frequency(metrics.dominant_frequency_hz))
        cols[1].metric("Tremor candidate", format_frequency(metrics.tremor_frequency_hz))

        st.progress(
            float(np.clip(metrics.confidence, 0.0, 1.0)),
            text=f"Confidence — {metrics.confidence * 100:0.0f}%",
        )
        st.caption(f"Amplitude score: {metrics.amplitude_score:0.1f}/100")

        if metrics.is_tremor_candidate:
            st.success(metrics.classification)
        elif metrics.classification == td.CLASS_COLLECTING:
            st.info(metrics.classification)
        else:
            st.warning(metrics.classification)

        st.image(
            render_waveform_image(metrics.waveform),
            channels="RGB",
            use_container_width=True,
        )


def run_capture_loop(
    *,
    camera_source: str,
    camera_index: int,
    points: tuple[str, ...],
    window_seconds: float,
    config: td.TremorAnalysisConfig,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    frame_slot,
    metrics_slot,
    status_slot,
) -> None:
    if not td.load_runtime_dependencies():
        status_slot.error("OpenCV / MediaPipe failed to load.")
        st.session_state.running = False
        return

    cv2 = td.cv2
    mp = td.mp

    try:
        capture = td.open_frame_source(camera_source, camera_index)
    except RuntimeError as exc:
        status_slot.error(str(exc))
        st.session_state.running = False
        return

    buffers = ensure_buffers(window_seconds, points)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as hands:
            while st.session_state.running:
                ok, camera_frame = capture.read()
                if not ok:
                    status_slot.error("Camera frame read failed.")
                    break
                frame = camera_frame.color
                depth_frame = camera_frame.depth
                now = time.perf_counter()

                for buffer in buffers.values():
                    buffer.prune(now)

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                hand_landmarks = td.choose_best_hand(results)
                hand_present = hand_landmarks is not None
                if hand_present and not st.session_state.was_hand_present:
                    st.session_state.hand_present_since = now
                if not hand_present:
                    st.session_state.hand_present_since = None
                st.session_state.was_hand_present = hand_present
                settled = (
                    st.session_state.hand_present_since is not None
                    and now - st.session_state.hand_present_since >= SETTLING_SECONDS
                )

                if hand_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    for point in points:
                        normalized_point, quality = td.normalized_motion_point(
                            hand_landmarks, point, depth_frame,
                        )
                        buffers[point].add(td.MotionSample(
                            timestamp=now,
                            x=float(normalized_point[0]),
                            y=float(normalized_point[1]),
                            quality=quality,
                            z=float(normalized_point[2]) if normalized_point.size >= 3 else None,
                        ))

                metrics_by_point = {}
                for point in points:
                    timestamps, positions, qualities = buffers[point].as_arrays()
                    metrics_by_point[point] = td.analyze_tremor(
                        timestamps=timestamps,
                        positions=positions,
                        qualities=qualities,
                        min_freq=config.measurement_min_freq,
                        max_freq=config.measurement_max_freq,
                        tremor_min_freq=config.tremor_min_freq,
                        tremor_max_freq=config.tremor_max_freq,
                        min_signal_seconds=config.min_signal_seconds,
                        min_confidence=config.min_confidence,
                        min_amplitude=config.min_amplitude,
                        min_tremor_power_ratio=config.min_tremor_power_ratio,
                        classification_enabled=settled,
                    )
                st.session_state.last_metrics = metrics_by_point

                display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(display, channels="RGB", use_container_width=True)

                # show first tracked point metrics
                first_point = points[0]
                render_metrics(metrics_slot, metrics_by_point[first_point], first_point)
                status_slot.caption("● recording")
    finally:
        capture.release()
        status_slot.caption("○ idle")


def main() -> None:
    st.set_page_config(page_title="NeuroTrack Tremor Dashboard", layout="wide")
    init_state()

    st.title("NeuroTrack Tremor Dashboard")
    st.caption("Live webcam tremor analysis. Prototype only — not a medical diagnostic device.")

    with st.sidebar:
        st.subheader("Capture")
        camera_source = st.selectbox("Source", ("webcam", "oak"), index=0)
        camera_index = st.number_input("Camera index", min_value=0, max_value=8, value=0, step=1)
        points = st.multiselect("Tracked points", TRACKED_POINTS, default=list(DEFAULT_TRACKED_POINTS))
        if not points:
            st.warning("Select at least one tracked point.")
            points = [TRACKED_POINTS[0]]
        selected_points = tuple(points)
        window_seconds = st.slider("Analysis window (s)", 5.0, 40.0, 20.0, 1.0)

        st.subheader("Classifier")
        preset_name = st.selectbox("Preset", tuple(PRESETS), index=0)
        preset = PRESETS[preset_name]

        config = td.TremorAnalysisConfig(
            min_signal_seconds=float(preset["min_signal_seconds"]),
            min_confidence=float(preset["min_confidence"]),
            min_amplitude=float(preset["min_amplitude"]),
            agreement_count=int(preset["agreement_count"]),
        )

        st.subheader("MediaPipe thresholds")
        min_detection_confidence = st.slider("Detection", 0.1, 0.95, 0.7, 0.05)
        min_tracking_confidence = st.slider("Tracking", 0.1, 0.95, 0.7, 0.05)

        st.divider()
        start_col, reset_col = st.columns(2)
        if start_col.button(
            "Stop" if st.session_state.running else "Start",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.running = not st.session_state.running
            if st.session_state.running and st.session_state.buffers is None:
                reset_buffers(window_seconds, selected_points)
            st.rerun()
        if reset_col.button("Reset buffer", use_container_width=True):
            reset_buffers(window_seconds, selected_points)
            st.rerun()

    feed_col, metrics_col = st.columns([3, 2])
    with feed_col:
        st.subheader("Live feed")
        frame_slot = st.empty()
        status_slot = st.empty()
    with metrics_col:
        st.subheader("Analysis")
        metrics_slot = st.empty()

    if st.session_state.running:
        run_capture_loop(
            camera_source=camera_source,
            camera_index=int(camera_index),
            points=selected_points,
            window_seconds=float(window_seconds),
            config=config,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            frame_slot=frame_slot,
            metrics_slot=metrics_slot,
            status_slot=status_slot,
        )
    else:
        frame_slot.info("Press **Start** in the sidebar to begin webcam capture.")
        status_slot.caption("○ idle")
        metrics_slot.caption("Metrics will appear once capture starts.")


if __name__ == "__main__":
    main()