"""Save the actual selected hull image, not a crop from a different timestamp."""

import json
import math
from pathlib import Path

from hauling.vision.plate import PlateCandidate


class HullEvidence:
    def __init__(self, directory: Path):
        self.directory = directory
        self.pending = {}
        self.records = []

    def remember(self, frame, detection, candidate: PlateCandidate):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = candidate.bbox_xyxy
        vx1, vy1, vx2, vy2 = detection.bbox_xyxy
        left, top = max(0, math.floor(vx1)), max(0, math.floor(vy1))
        right, bottom = min(width, math.ceil(vx2)), min(height, math.ceil(vy2))
        self.pending[detection.track_id] = (
            candidate, frame[y1:y2, x1:x2].copy(),
            frame[top:bottom, left:right].copy(), (left, top, right, bottom),
        )

    def save(self, event_id: str, track_id: int):
        import cv2

        item = self.pending.pop(track_id, None)
        if item is None:
            return False
        candidate, panel, context, vehicle_box = item
        self.directory.mkdir(parents=True, exist_ok=True)
        prefix = f"event_{int(event_id):04d}_track_{track_id}"
        crop_name, context_name = f"{prefix}.jpg", f"{prefix}_context.jpg"
        x1, y1, x2, y2 = candidate.bbox_xyxy
        left, top, _, _ = vehicle_box
        cv2.rectangle(context, (x1 - left, y1 - top), (x2 - left, y2 - top), (255, 255, 0), 2)
        if not cv2.imwrite(str(self.directory / crop_name), panel):
            raise OSError("Cannot save hull-panel crop")
        if not cv2.imwrite(str(self.directory / context_name), context):
            raise OSError("Cannot save hull-panel context")
        self.records.append({
            "event_id": event_id, "track_id": track_id,
            "timestamp_s": candidate.timestamp_s,
            "bbox_frame_xyxy": list(candidate.bbox_xyxy),
            "vehicle_bbox_frame_xyxy": list(vehicle_box),
            "source": candidate.source, "score": candidate.confidence,
            "text": candidate.text, "text_confidence": candidate.text_confidence,
            "text_status": candidate.text_status,
            "ocr_observations": [{"timestamp_s": p.timestamp_s, "text": p.text,
                                  "status": p.text_status, "confidence": p.confidence}
                                 for p in candidate.observations],
            "characters": [{"text": c.text, "confidence": c.confidence,
                            "bbox_frame_xyxy": list(c.bbox_xyxy)} for c in candidate.characters],
            "score_is_probability": False if candidate.source == "reference_appearance" else None,
            "crop": crop_name, "context": context_name,
        })
        return True

    def finalize(self, track_id: int, candidate: PlateCandidate) -> None:
        """Attach consensus to the stored source pixels without recropping a later frame."""
        pending = self.pending.get(track_id)
        if pending is not None:
            original, *images = pending
            if (original.timestamp_s, original.bbox_xyxy) != (candidate.timestamp_s, candidate.bbox_xyxy):
                raise ValueError("Hull evidence pixels do not match the selected source frame")
            self.pending[track_id] = (candidate, *images)

    def close(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "manifest.json").write_text(
            json.dumps(self.records, indent=2) + "\n", encoding="utf-8")
