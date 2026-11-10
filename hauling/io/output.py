"""The CSV contract is kept here, separate from inference."""

import csv
import json
from pathlib import Path

from hauling.config import PipelineConfig
from hauling.engine.state import VehicleEvent

CSV_HEADERS = (
    "event_id", "source_video", "track_id", "event_timestamp_s", "event_frame_index",
    "travel_direction", "vehicle_confidence", "speed_start_timestamp_s",
    "speed_end_timestamp_s", "elapsed_seconds", "distance_meter", "speed_kmh",
    "speed_status", "speed_limit_kmh", "is_overspeed", "plate_detected",
    "plate_confidence", "plate_bbox_xyxy", "plate_timestamp_s", "plate_text",
    "measurement_valid", "notes",
    "vehicle_class_id", "vehicle_class", "plate_text_confidence",
    "plate_text_status", "violation_status", "violation_reason",
)


def number(value: float | None, decimals: int = 6) -> str:
    return "" if value is None else f"{value:.{decimals}f}"


def event_record(event: VehicleEvent, config: PipelineConfig, source: str, event_id: str) -> dict:
    track, speed = event.track, event.measurement
    plate = track.plate
    notes = []
    if plate is not None and any(p.timestamp_s > (track.finish_s if track.finish_s is not None else event.timestamp_s) for p in plate.observations):
        notes.append("ocr_lookahead_used")
    if plate is not None and plate.timestamp_s > (track.finish_s if track.finish_s is not None else event.timestamp_s):
        notes.append("ocr_source_after_finish")
    if track.ocr_window_truncated:
        notes.append("ocr_window_truncated")
    if track.vehicle_class_source != "detector":
        notes.append(f"vehicle_class_source={track.vehicle_class_source}")
    if plate is not None and plate.source != "yolo":
        notes.append(f"hull_source={plate.source}")
    if track.direction not in {"Unknown", config.travel_direction} and config.travel_direction != "Unknown":
        notes.append("direction_mismatch")
    return dict(zip(CSV_HEADERS, (
        event_id, source, track.track_id, number(event.timestamp_s), event.frame_index,
        track.direction, number(track.vehicle_confidence), number(track.start_s),
        number(track.finish_s), number(speed.elapsed_seconds), number(config.distance_meter),
        number(speed.speed_kmh, 2), speed.status, number(config.speed_limit_kmh),
        str(speed.is_overspeed).lower(), str(plate is not None).lower(),
        number(plate.confidence if plate else None),
        json.dumps(plate.bbox_xyxy, separators=(",", ":")) if plate else "",
        number(plate.timestamp_s if plate else None), plate.text if plate else "", str(speed.valid).lower(),
        ";".join(notes),
        track.vehicle_class_id if track.vehicle_class_id is not None else "", track.vehicle_class,
        number(plate.text_confidence if plate else None),
        plate.text_status if plate else "",
        track.violation["violation_status"], track.violation["violation_reason"] or "",
    )))


class EventWriter:
    def __init__(self, path: Path):
        self.handle = path.open("x", encoding="utf-8", newline="")
        try:
            self.writer = csv.DictWriter(self.handle, fieldnames=CSV_HEADERS)
            self.writer.writeheader()
            self.handle.flush()
        except BaseException:
            self.handle.close()
            raise
        self.track_ids: set[int] = set()

    def write(self, record: dict) -> None:
        track_id = record["track_id"]
        if track_id in self.track_ids:
            raise ValueError("Duplicate event for track")
        self.writer.writerow(record)
        self.handle.flush()
        self.track_ids.add(track_id)

    def close(self) -> None:
        self.handle.close()
