"""
Sentry - Streamlit Dashboard
Real-time weapon detection dashboard with webcam and image upload support.

Run with: streamlit run src/app.py
"""

import sys
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from threat_monitor import ThreatMonitor
from utils import draw_detection_box, draw_alert_overlay

# Page configuration
st.set_page_config(
    page_title="Sentry - Weapon Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Red Alert state
ALERT_CSS = """
<style>
.red-alert {
    background-color: #ff0000 !important;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}
.red-alert h1 {
    color: white !important;
    font-size: 2.5rem !important;
    margin: 0 !important;
}
.status-safe {
    background-color: #28a745;
    padding: 10px 20px;
    border-radius: 5px;
    color: white;
    text-align: center;
}
.status-alert {
    background-color: #dc3545;
    padding: 10px 20px;
    border-radius: 5px;
    color: white;
    text-align: center;
    animation: pulse 1s infinite;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
"""

# Available models
AVAILABLE_MODELS = {
    "Sentry (Custom Trained)": "models/sentry_best.pt",
    "YOLOv8-Nano (Base)": "notebooks/yolov8n.pt",
    "YOLOv11-Nano (Base)": "notebooks/yolo11n.pt"
}
DEFAULT_MODEL = "Sentry (Custom Trained)"


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Load and cache YOLO model."""
    return YOLO(model_path)


class VideoProcessor(VideoProcessorBase):
    """Video processor for WebRTC live detection."""

    def __init__(self):
        self.model = None
        self.conf_threshold = 0.25
        self.result_queue = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process each video frame."""
        img = frame.to_ndarray(format="bgr24")

        if self.model is not None:
            # Run inference
            results = self.model.predict(img, conf=self.conf_threshold, verbose=False)[0]
            boxes = results.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]

                    # Draw bounding box
                    x1, y1, x2, y2 = xyxy
                    color = (0, 0, 255) if cls_name in ['gun', 'knife'] else (255, 165, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{cls_name}: {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def init_session_state():
    """Initialize session state variables."""
    if 'threat_monitor' not in st.session_state:
        st.session_state.threat_monitor = ThreatMonitor(persistence_frames=5)
    if 'incident_log' not in st.session_state:
        st.session_state.incident_log = []
    if 'detection_stats' not in st.session_state:
        st.session_state.detection_stats = {'gun': 0, 'knife': 0, 'person_with_mask': 0}
    if 'total_frames' not in st.session_state:
        st.session_state.total_frames = 0
    if 'alert_active' not in st.session_state:
        st.session_state.alert_active = False


def process_frame_streamlit(frame: np.ndarray, model: YOLO, conf_threshold: float) -> tuple:
    """
    Process a frame and return results for Streamlit display.

    Returns:
        Tuple of (processed_frame, detections_list, alert_triggered)
    """
    # Run inference
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]

    detections = []
    boxes = results.boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            detections.append({
                'class': cls_name,
                'confidence': conf,
                'box': xyxy
            })

            # Draw bounding box
            frame = draw_detection_box(frame, xyxy, cls_name, conf)

    # Update threat monitor
    detected_classes = [d['class'] for d in detections]
    alert_triggered = st.session_state.threat_monitor.update(detected_classes)

    # Draw alert overlay if triggered
    if alert_triggered:
        threat_classes = st.session_state.threat_monitor.get_threat_classes()
        message = f"THREAT: {', '.join(threat_classes).upper()}"
        frame = draw_alert_overlay(frame, message)

    return frame, detections, alert_triggered


def add_to_incident_log(detections: list, frame_num: int):
    """Add detections to incident log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for det in detections:
        entry = {
            'Timestamp': timestamp,
            'Class': det['class'],
            'Confidence': f"{det['confidence']:.2%}",
            'Frame': frame_num
        }
        st.session_state.incident_log.append(entry)

        # Update stats
        if det['class'] in st.session_state.detection_stats:
            st.session_state.detection_stats[det['class']] += 1

    # Keep only last 100 entries
    if len(st.session_state.incident_log) > 100:
        st.session_state.incident_log = st.session_state.incident_log[-100:]


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.title("🛡️ Sentry Controls")
        st.divider()

        # Mode selector
        mode = st.radio(
            "Detection Mode",
            ["📷 Image Upload", "🎥 Webcam"],
            index=0
        )

        st.divider()

        # Model selector
        model_name = st.selectbox(
            "Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            help="Select detection model"
        )
        model_path = AVAILABLE_MODELS[model_name]

        st.divider()

        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detection"
        )

        # Persistence frames
        persistence = st.slider(
            "Alert Persistence (frames)",
            min_value=1,
            max_value=10,
            value=5,
            help="Consecutive frames needed to trigger alert"
        )

        # Update threat monitor persistence if changed
        if st.session_state.threat_monitor.persistence_frames != persistence:
            st.session_state.threat_monitor = ThreatMonitor(persistence_frames=persistence)

        st.divider()

        # Reset button
        if st.button("🔄 Reset Statistics", use_container_width=True):
            st.session_state.incident_log = []
            st.session_state.detection_stats = {'gun': 0, 'knife': 0, 'person_with_mask': 0}
            st.session_state.total_frames = 0
            st.session_state.threat_monitor.reset()
            st.rerun()

        st.divider()

        # Model info
        st.caption("**Model Info**")
        st.caption(f"Model: `{model_name}`")
        st.caption(f"Path: `{model_path}`")
        st.caption("Classes: gun, knife, person_with_mask")

    return mode, conf_threshold, persistence, model_path


def render_statistics():
    """Render statistics panel."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Detections", sum(st.session_state.detection_stats.values()))

    with col2:
        st.metric("🔫 Guns", st.session_state.detection_stats.get('gun', 0))

    with col3:
        st.metric("🔪 Knives", st.session_state.detection_stats.get('knife', 0))

    with col4:
        st.metric("😷 Masked", st.session_state.detection_stats.get('person_with_mask', 0))


def render_alert_status(alert_active: bool):
    """Render alert status banner."""
    if alert_active:
        st.markdown("""
        <div class="red-alert">
            <h1>⚠️ THREAT DETECTED ⚠️</h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-safe">
            <strong>✅ System Monitoring - No Threats Detected</strong>
        </div>
        """, unsafe_allow_html=True)


def render_incident_log():
    """Render incident log dataframe."""
    st.subheader("📋 Incident Log")

    if st.session_state.incident_log:
        df = pd.DataFrame(st.session_state.incident_log[-50:][::-1])  # Last 50, newest first
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("No detections recorded yet.")


def image_upload_mode(model: YOLO, conf_threshold: float):
    """Handle image upload mode."""
    st.subheader("📷 Image Upload Detection")

    uploaded_file = st.file_uploader(
        "Upload an image for analysis",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Drag and drop or click to upload"
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is not None:
            st.session_state.total_frames += 1

            # Process frame
            processed_frame, detections, alert = process_frame_streamlit(
                frame, model, conf_threshold
            )

            # Update session state
            st.session_state.alert_active = alert

            # Add to incident log
            if detections:
                add_to_incident_log(detections, st.session_state.total_frames)

            # Display results
            col1, col2 = st.columns([2, 1])

            with col1:
                # Convert BGR to RGB for display
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(display_frame, caption=f"Detection Results ({len(detections)} objects)", width="stretch")

            with col2:
                st.subheader("Detection Results")
                if detections:
                    for det in detections:
                        color = "🔴" if det['class'] in ['gun', 'knife'] else "🔵"
                        st.write(f"{color} **{det['class']}**: {det['confidence']:.2%}")
                else:
                    st.info("No objects detected")

                # Alert status
                st.divider()
                if alert:
                    st.error("⚠️ **THREAT ALERT ACTIVE**")
                else:
                    st.success("✅ No active threats")


def webcam_mode(model: YOLO, conf_threshold: float):
    """Handle webcam mode with both snapshot and live video options."""
    st.subheader("🎥 Camera Detection")

    # Sub-mode selector
    camera_mode = st.radio(
        "Camera Mode",
        ["📸 Snapshot", "🔴 Live Video"],
        horizontal=True,
        help="Snapshot: Take photos for analysis. Live Video: Real-time detection stream."
    )

    if camera_mode == "📸 Snapshot":
        # Snapshot mode using st.camera_input
        st.info("📷 Click the camera button to capture a photo for analysis.")

        camera_photo = st.camera_input("Take a picture for threat detection")

        if camera_photo is not None:
            # Read image from camera input
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                st.session_state.total_frames += 1

                # Process frame
                processed_frame, detections, alert = process_frame_streamlit(
                    frame, model, conf_threshold
                )

                # Update session state
                st.session_state.alert_active = alert

                # Add to incident log
                if detections:
                    add_to_incident_log(detections, st.session_state.total_frames)

                # Display results
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Convert BGR to RGB for display
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st.image(display_frame, caption=f"Detection Results ({len(detections)} objects)", width="stretch")

                with col2:
                    st.subheader("Detection Results")
                    if detections:
                        for det in detections:
                            color = "🔴" if det['class'] in ['gun', 'knife'] else "🔵"
                            st.write(f"{color} **{det['class']}**: {det['confidence']:.2%}")
                    else:
                        st.info("No objects detected")

                    # Alert status
                    st.divider()
                    if alert:
                        st.error("⚠️ **THREAT ALERT ACTIVE**")
                    else:
                        st.success("✅ No active threats")

    else:
        # Live video mode using WebRTC
        st.info("🔴 Live video detection. Click START to begin streaming from your camera.")

        # RTC Configuration for STUN servers (needed for WebRTC)
        rtc_configuration = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }

        # Create WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="weapon-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Pass model and settings to the video processor
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.model = model
            webrtc_ctx.video_processor.conf_threshold = conf_threshold

        # Display status
        if webrtc_ctx.state.playing:
            st.success("🔴 Live detection active")
        else:
            st.warning("⏸️ Click START above to begin live detection")


def main():
    """Main application entry point."""
    # Inject custom CSS
    st.markdown(ALERT_CSS, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Render sidebar and get settings (including model selection)
    mode, conf_threshold, persistence, model_path = render_sidebar()

    # Load selected model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.info(f"Please ensure the model exists at: `{model_path}`")
        return

    # Main content area
    st.title("🛡️ Sentry - Weapon Detection System")

    # Alert status banner
    render_alert_status(st.session_state.alert_active)

    st.divider()

    # Statistics
    render_statistics()

    st.divider()

    # Main detection area
    if "Image" in mode:
        image_upload_mode(model, conf_threshold)
    else:
        webcam_mode(model, conf_threshold)

    st.divider()

    # Incident log
    render_incident_log()


if __name__ == "__main__":
    main()
