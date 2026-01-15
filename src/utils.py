"""
Sentry - Utility Functions
Helper functions for detection visualization and logging.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path


# Class colors in BGR format (OpenCV uses BGR, not RGB)
CLASS_COLORS = {
    "gun": (0, 0, 255),           # Red
    "knife": (0, 165, 255),       # Orange
    "person_with_mask": (255, 0, 0)  # Blue
}

# Default color for unknown classes
DEFAULT_COLOR = (0, 255, 0)  # Green


def get_class_color(class_name: str) -> tuple:
    """
    Get BGR color for a given class name.

    Args:
        class_name: Name of the detected class

    Returns:
        BGR color tuple
    """
    return CLASS_COLORS.get(class_name.lower(), DEFAULT_COLOR)


def draw_detection_box(frame: np.ndarray, box, class_name: str,
                       confidence: float, color: tuple = None) -> np.ndarray:
    """
    Draw a bounding box with label on the frame.

    Args:
        frame: Input image/frame
        box: Bounding box coordinates [x1, y1, x2, y2]
        class_name: Name of detected class
        confidence: Detection confidence score
        color: Optional BGR color tuple (auto-selected if None)

    Returns:
        Frame with drawn bounding box
    """
    if color is None:
        color = get_class_color(class_name)

    x1, y1, x2, y2 = map(int, box)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Create label with class name and confidence
    label = f"{class_name}: {confidence:.2f}"

    # Get text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw background rectangle for label
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), color, -1)

    # Draw label text
    cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)

    return frame


def draw_alert_overlay(frame: np.ndarray, message: str = "THREAT DETECTED") -> np.ndarray:
    """
    Draw red alert overlay on frame when threat is confirmed.

    Args:
        frame: Input image/frame
        message: Alert message to display

    Returns:
        Frame with red alert overlay
    """
    h, w = frame.shape[:2]

    # Draw red border
    border_thickness = 10
    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)

    # Draw alert banner at top
    banner_height = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw alert text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(message, font, font_scale, thickness)
    text_x = (w - text_width) // 2
    text_y = (banner_height + text_height) // 2
    cv2.putText(frame, message, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return frame


def save_screenshot(frame: np.ndarray, output_dir: str = "data/screenshots",
                    prefix: str = "threat") -> str:
    """
    Save a screenshot of the current frame.

    Args:
        frame: Frame to save
        output_dir: Directory to save screenshots
        prefix: Filename prefix

    Returns:
        Path to saved screenshot
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, frame)
    return filepath


def log_detection(class_name: str, confidence: float, frame_num: int = None,
                  log_file: str = "data/detection_log.txt") -> None:
    """
    Log detection event to file.

    Args:
        class_name: Detected class name
        confidence: Detection confidence
        frame_num: Optional frame number
        log_file: Path to log file
    """
    # Ensure directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frame_info = f" [Frame {frame_num}]" if frame_num is not None else ""

    log_entry = f"{timestamp}{frame_info} - Detected: {class_name} (conf: {confidence:.3f})\n"

    with open(log_file, "a") as f:
        f.write(log_entry)


def format_fps(fps: float) -> str:
    """Format FPS for display."""
    return f"FPS: {fps:.1f}"


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw FPS counter on frame.

    Args:
        frame: Input frame
        fps: Frames per second value

    Returns:
        Frame with FPS counter
    """
    text = format_fps(fps)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10, 30), font, 0.7, (0, 255, 0), 2)
    return frame
