"""
Sentry - Threat Persistence Monitor
Implements threat detection persistence to prevent false alerts.
Only triggers alert after weapon detected in N consecutive frames.
"""

from collections import deque
from typing import List, Set


# Threat classes (weapons that trigger alerts)
THREAT_CLASSES = {"gun", "knife"}


class ThreatMonitor:
    """
    Monitors threat detections across frames to prevent false positives.

    Alert is triggered only when a threat (gun/knife) is detected
    in a specified number of consecutive frames.
    """

    def __init__(self, persistence_frames: int = 5):
        """
        Initialize the threat monitor.

        Args:
            persistence_frames: Number of consecutive frames with detection
                               required to trigger alert (default: 5)
        """
        self.persistence_frames = persistence_frames
        self.detection_buffer: deque = deque(maxlen=persistence_frames)
        self.alert_active = False
        self._last_threat_classes: Set[str] = set()

    def update(self, detected_classes: List[str]) -> bool:
        """
        Update the monitor with current frame's detections.

        Args:
            detected_classes: List of class names detected in current frame

        Returns:
            True if threat alert should be triggered, False otherwise
        """
        # Check if any threat class is present in detections
        current_threats = set(c.lower() for c in detected_classes) & THREAT_CLASSES
        has_threat = len(current_threats) > 0

        # Add to buffer
        self.detection_buffer.append(has_threat)

        # Update last detected threat classes
        if has_threat:
            self._last_threat_classes = current_threats

        # Check if buffer is full and all frames have threats
        if len(self.detection_buffer) == self.persistence_frames:
            self.alert_active = all(self.detection_buffer)
        else:
            self.alert_active = False

        return self.alert_active

    def get_alert_status(self) -> bool:
        """
        Get current alert status.

        Returns:
            True if alert is active, False otherwise
        """
        return self.alert_active

    def get_threat_classes(self) -> Set[str]:
        """
        Get the last detected threat classes that triggered the alert.

        Returns:
            Set of threat class names
        """
        return self._last_threat_classes if self.alert_active else set()

    def get_persistence_progress(self) -> tuple:
        """
        Get current persistence buffer progress.

        Returns:
            Tuple of (current_count, required_count)
        """
        threat_count = sum(1 for d in self.detection_buffer if d)
        return (threat_count, self.persistence_frames)

    def reset(self) -> None:
        """Reset the monitor state."""
        self.detection_buffer.clear()
        self.alert_active = False
        self._last_threat_classes.clear()

    def __repr__(self) -> str:
        progress = self.get_persistence_progress()
        return (f"ThreatMonitor(persistence={self.persistence_frames}, "
                f"progress={progress[0]}/{progress[1]}, alert={self.alert_active})")
