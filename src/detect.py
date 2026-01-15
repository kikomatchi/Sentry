"""
Sentry - Core Detection Script
Real-time weapon detection using YOLOv8.

Usage:
    python src/detect.py --source 0              # Webcam
    python src/detect.py --source video.mp4      # Video file
    python src/detect.py --source image.jpg      # Single image
    python src/detect.py --source folder/        # Image folder
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from threat_monitor import ThreatMonitor
from utils import (
    draw_detection_box,
    draw_alert_overlay,
    draw_fps,
    save_screenshot,
    log_detection,
    get_class_color
)


# Default model path
DEFAULT_MODEL_PATH = "models/sentry_best.pt"


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    """
    Load YOLOv8 model from file.

    Args:
        model_path: Path to model weights (.pt file)

    Returns:
        Loaded YOLO model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))
    print(f"Model loaded. Classes: {model.names}")
    return model


def process_frame(frame: np.ndarray, model: YOLO, conf_threshold: float,
                  threat_monitor: ThreatMonitor, frame_num: int = 0,
                  save_alerts: bool = True) -> tuple:
    """
    Process a single frame through the detection pipeline.

    Args:
        frame: Input frame (BGR)
        model: YOLO model
        conf_threshold: Confidence threshold
        threat_monitor: ThreatMonitor instance
        frame_num: Current frame number
        save_alerts: Whether to save screenshots on alert

    Returns:
        Tuple of (processed_frame, detection_count, alert_triggered)
    """
    # Run inference
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]

    # Extract detections
    detected_classes = []
    boxes = results.boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Get box coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            detected_classes.append(cls_name)

            # Draw bounding box
            frame = draw_detection_box(frame, xyxy, cls_name, conf)

            # Log detection
            log_detection(cls_name, conf, frame_num)

    # Update threat monitor
    alert_triggered = threat_monitor.update(detected_classes)

    # Draw alert overlay if triggered
    if alert_triggered:
        threat_classes = threat_monitor.get_threat_classes()
        message = f"THREAT: {', '.join(threat_classes).upper()}"
        frame = draw_alert_overlay(frame, message)

        # Save screenshot on new alert
        if save_alerts:
            screenshot_path = save_screenshot(frame)
            print(f"Alert screenshot saved: {screenshot_path}")

    return frame, len(detected_classes), alert_triggered


def run_detection_video(source, model: YOLO, conf_threshold: float = 0.25,
                        persistence_frames: int = 5, save_alerts: bool = True,
                        show_window: bool = True) -> None:
    """
    Run detection on video stream (webcam or file).

    Args:
        source: Video source (0 for webcam, or path to video file)
        model: YOLO model
        conf_threshold: Confidence threshold
        persistence_frames: Frames required for alert
        save_alerts: Save screenshots on alert
        show_window: Display video window
    """
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    # Get video properties
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source: {width}x{height} @ {fps_cap:.1f} FPS")

    # Initialize threat monitor
    threat_monitor = ThreatMonitor(persistence_frames=persistence_frames)

    frame_num = 0
    fps_calc = 0
    prev_time = time.time()
    alert_cooldown = 0  # Prevent saving duplicate screenshots

    print("Starting detection... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video or read error
                if isinstance(source, str):
                    print("End of video file.")
                break

            frame_num += 1

            # Process frame
            should_save = save_alerts and alert_cooldown == 0
            processed_frame, det_count, alert = process_frame(
                frame, model, conf_threshold, threat_monitor,
                frame_num, save_alerts=should_save
            )

            # Manage alert cooldown (save screenshot every N frames during alert)
            if alert:
                if alert_cooldown == 0:
                    alert_cooldown = 30  # Save every 30 frames during alert
                else:
                    alert_cooldown -= 1
            else:
                alert_cooldown = 0

            # Calculate FPS
            curr_time = time.time()
            fps_calc = 1.0 / (curr_time - prev_time) if curr_time != prev_time else fps_calc
            prev_time = curr_time

            # Draw FPS
            processed_frame = draw_fps(processed_frame, fps_calc)

            # Display
            if show_window:
                cv2.imshow("Sentry - Weapon Detection", processed_frame)

                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("Detection stopped by user.")
                    break

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    print(f"Processed {frame_num} frames.")


def run_detection_image(source: str, model: YOLO, conf_threshold: float = 0.25,
                        show_window: bool = True, save_output: bool = True) -> None:
    """
    Run detection on single image or folder of images.

    Args:
        source: Path to image or folder
        model: YOLO model
        conf_threshold: Confidence threshold
        show_window: Display result window
        save_output: Save annotated images
    """
    source_path = Path(source)

    if source_path.is_dir():
        # Process folder of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in source_path.iterdir()
                       if f.suffix.lower() in image_extensions]
        print(f"Found {len(image_files)} images in {source_path}")
    else:
        image_files = [source_path]

    # Initialize threat monitor (for demonstration, won't actually track across images)
    threat_monitor = ThreatMonitor(persistence_frames=1)  # Single frame mode

    for idx, img_path in enumerate(image_files):
        print(f"Processing [{idx+1}/{len(image_files)}]: {img_path.name}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Failed to load image, skipping.")
            continue

        # Process
        processed_frame, det_count, alert = process_frame(
            frame, model, conf_threshold, threat_monitor,
            frame_num=idx, save_alerts=False
        )

        print(f"  Detections: {det_count}")

        if save_output:
            output_path = source_path / "output" if source_path.is_dir() else source_path.parent / "output"
            output_path.mkdir(exist_ok=True)
            out_file = output_path / f"detected_{img_path.name}"
            cv2.imwrite(str(out_file), processed_frame)
            print(f"  Saved: {out_file}")

        if show_window:
            cv2.imshow("Sentry - Detection Result", processed_frame)
            key = cv2.waitKey(0) & 0xFF  # Wait for key press
            if key == ord('q') or key == 27:
                break

        # Reset monitor between images
        threat_monitor.reset()

    if show_window:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Sentry - Real-time Weapon Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source", "-s",
        default="0",
        help="Video source: 0 for webcam, path to video/image file, or folder"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLOv8 model weights"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--persistence", "-p",
        type=int,
        default=5,
        help="Number of consecutive frames for threat alert"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving alert screenshots"
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable display window (headless mode)"
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Determine source type
    source = args.source

    # Check if source is webcam index
    if source.isdigit():
        source = int(source)
        print(f"Using webcam: {source}")
        run_detection_video(
            source, model,
            conf_threshold=args.conf,
            persistence_frames=args.persistence,
            save_alerts=not args.no_save,
            show_window=not args.no_window
        )
    else:
        source_path = Path(source)
        if not source_path.exists():
            print(f"Error: Source not found: {source}")
            sys.exit(1)

        # Check if video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        if source_path.suffix.lower() in video_extensions:
            print(f"Processing video file: {source}")
            run_detection_video(
                source, model,
                conf_threshold=args.conf,
                persistence_frames=args.persistence,
                save_alerts=not args.no_save,
                show_window=not args.no_window
            )
        else:
            # Image or folder
            print(f"Processing image(s): {source}")
            run_detection_image(
                source, model,
                conf_threshold=args.conf,
                show_window=not args.no_window,
                save_output=not args.no_save
            )


if __name__ == "__main__":
    main()
