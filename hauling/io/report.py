"""Optional per-event report with vehicle photos from final-area contact."""

import json
import math
from pathlib import Path


class TrackReport:
    def __init__(self, directory: Path):
        self.directory = directory
        self.pending = {}
        self.records = []
        self.photos_saved = 0

    def remember(self, frame, detection, event) -> None:
        """Keep source pixels at event contact while bounded OCR completes."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox_xyxy
        box = (max(0, math.floor(x1)), max(0, math.floor(y1)),
               min(width, math.ceil(x2)), min(height, math.ceil(y2)))
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            raise ValueError("Vehicle report crop is outside the source frame")
        self.pending[event.track.track_id] = (
            frame[top:bottom, left:right].copy(), box, event.timestamp_s, event.frame_index,
        )

    def save(self, record: dict) -> None:
        """Attach finalized CSV values only after the event was persisted."""
        import cv2

        entry = dict(record, photo=None)
        self.records.append(entry)
        item = self.pending.pop(record["track_id"], None)
        if item is None:
            return
        image, box, timestamp_s, frame_index = item
        name = f"tracks/event_{int(record['event_id']):04d}_track_{int(record['track_id'])}.jpg"
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            saved = cv2.imwrite(str(path), image)
        except Exception as error:
            raise OSError("Cannot save report vehicle photo") from error
        if not saved:
            raise OSError("Cannot save report vehicle photo")
        entry.update(photo=name, photo_bbox_frame_xyxy=list(box),
                     photo_timestamp_s=timestamp_s, photo_frame_index=frame_index)
        self.photos_saved += 1

    def close(self, summary: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "tracks.json").write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        lines = ["# Hauling vehicle report", "", f"Run status: {summary['status']}",
                 f"Final-area events: {len(self.records)}", "",
                 "Only persisted final-area events are included. Photos use the original",
                 "vehicle crop at event contact; OCR evidence may use a different timestamp.", ""]
        for entry in self.records:
            lines.extend([
                f"## Track {entry['track_id']} (event {entry['event_id']})", "",
                f"Event time: {entry['event_timestamp_s']} s; frame: {entry['event_frame_index']}",
                f"Vehicle: {entry['vehicle_class']}",
                f"Speed: {entry['speed_kmh'] or 'unavailable'} km/h; status: {entry['violation_status']}",
                f"Hull prediction: {entry['plate_text'] or 'unread'}", "",
            ])
            if entry["photo"]:
                lines.extend([f"![Track {entry['track_id']}]({entry['photo']})", ""])
            else:
                lines.extend(["Vehicle photo unavailable; see run warnings.", ""])
        if not self.records:
            lines.extend(["No final-area events were persisted.", ""])
        (self.directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
