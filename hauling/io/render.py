"""Draw validation overlays and write a local MP4, without a GUI."""

from pathlib import Path
from collections import deque
from copy import copy
from dataclasses import dataclass
from typing import Any

from hauling.config import PipelineConfig, Polygon
from hauling.vision.detection import Detection
from hauling.utils.geometry import centroid, contains
from hauling.vision.plate import PlateCandidate
from hauling.engine.state import TrackEngine

AREA_COLORS = {"main": (255, 180, 0), "start": (0, 220, 0), "finish": (0, 220, 255), "final": (220, 0, 220)}


@dataclass
class BufferedFrame:
    frame: Any
    detections: list[Detection]
    engine: TrackEngine
    timestamp: float
    plates: dict[int, PlateCandidate]


class FrameBuffer:
    """Delay a few frames while preserving each frame's speed/type state."""
    def __init__(self, delay_frames: int):
        self.delay_frames = delay_frames
        self.frames = deque()

    def push(self, frame, detections, engine, timestamp, plates) -> None:
        snapshot = copy(engine)
        snapshot.tracks = {track: copy(state) for track, state in engine.tracks.items()}
        self.frames.append(BufferedFrame(frame, list(detections), snapshot, timestamp, dict(plates)))

    def ready(self, force: bool = False) -> list[BufferedFrame]:
        count = len(self.frames) if force else max(0, len(self.frames) - self.delay_frames)
        return [self.frames.popleft() for _ in range(count)]


def annotate(frame, detections: list[Detection], engine: TrackEngine,
             areas: dict[str, Polygon], config: PipelineConfig, timestamp: float,
             plate_enabled: bool, current_plates: dict[int, PlateCandidate] | None = None):
    import cv2
    import numpy as np

    image = frame.copy()
    for name, polygon in areas.items():
        points = np.rint(polygon).astype(np.int32)
        color = AREA_COLORS[name]
        cv2.polylines(image, [points], True, color, 2)
        px, py = points[0]
        cv2.putText(image, name, (max(0, int(px)), max(18, int(py) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for detection in detections:
        if detection.track_id not in engine.tracks and not contains(areas["main"], centroid(detection.bbox_xyxy)):
            continue
        x1, y1, x2, y2 = (round(x) for x in detection.bbox_xyxy)
        state = engine.tracks.get(detection.track_id)
        color = (0, 220, 0)
        label = f"{detection.class_name} ID={detection.track_id if detection.track_id is not None else 'unassigned'}"
        if state:
            measurement = state.measurement(config)
            if measurement.valid:
                label += f" {measurement.speed_kmh:.2f} km/h"
                if measurement.is_overspeed:
                    label += " OVERSPEED"
                    color = (0, 0, 255)
            else:
                label += f" {measurement.status}"
            plate = (current_plates or {}).get(state.track_id)
            if state.finish_s == timestamp and state.plate and state.plate.text:
                label += f" hull={state.plate.text}"
            if plate:
                px1, py1, px2, py2 = plate.bbox_xyxy
                cv2.rectangle(image, (px1, py1), (px2, py2), (255, 255, 0), 2)
                text = state.plate.text if state.plate else plate.text
                cv2.putText(image, f"hull={text or 'unread'} ID={state.track_id} conf={plate.confidence:.2f}",
                            (px1, max(18, py1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 0), 1)
                for index, character in enumerate(plate.characters):
                    cx1, cy1, cx2, cy2 = character.bbox_xyxy
                    cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (0, 165, 255), 1)
                    glyph = state.plate.characters[index].text if state.plate and len(state.plate.characters) == len(plate.characters) else character.text
                    cv2.putText(image, glyph, (cx1, max(12, cy2 + 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (round((x1 + x2) / 2), round((y1 + y2) / 2)), 3, color, -1)
        cv2.putText(image, label, (max(0, x1), max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    header = f"t={timestamp:.2f}s | limit={config.speed_limit_kmh:g} km/h"
    if not plate_enabled:
        header += " | hull detection DISABLED"
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 850), 30), (0, 0, 0), -1)
    cv2.putText(image, header, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return image


class AnnotatedVideo:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        import cv2

        self.path = path
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.frames_written = 0
        if not self.writer.isOpened():
            self.writer.release()
            raise OSError("Cannot open MP4 writer")

    def write(self, frame) -> None:
        self.writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        self.writer.release()

    def verify(self) -> bool:
        import cv2

        capture = cv2.VideoCapture(str(self.path))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, _ = capture.read()
            return capture.isOpened() and ok and count == self.frames_written
        finally:
            capture.release()
