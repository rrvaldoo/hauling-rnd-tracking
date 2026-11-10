"""Opt-in per-detection diagnostics, including observations rejected by the ROI."""

import json
import logging
from pathlib import Path

from hauling.utils.geometry import centroid, contains, crossing_fraction

LOGGER = logging.getLogger("hauling.io.trace")


class DetectionTrace:
    def __init__(self, path: Path, areas, fps: float, max_gap_frames: int):
        self.handle = path.open("x", encoding="utf-8")
        self.areas = areas
        self.fps = fps
        self.max_gap_frames = max_gap_frames
        self.last_observed = {}
        self.previous_ids = set()
        self.frames = self.detections = 0
        self.emit({"type": "metadata", "fps": fps, "areas": areas,
                   "max_crossing_gap_frames": max_gap_frames,
                   "timestamp_source": "zero_based_frame_index / fps",
                   "detection_scope": "all post-confidence/NMS boxes, including unassigned and outside main"})

    def emit(self, record):
        self.handle.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def begin_frame(self, index, timestamp, detections, count):
        ids = {d.track_id for d in detections if d.track_id is not None}
        missing = sorted(self.previous_ids - ids)
        self.emit({"type": "frame", "frame_index": index, "timestamp_s": timestamp,
                   "detector_count": count, "tracked_ids": sorted(ids),
                   "missing_since_previous_frame": missing,
                   "unassigned_count": sum(d.track_id is None for d in detections)})
        LOGGER.info("frame_detection frame=%s t=%.6f detections=%s ids=%s missing=%s",
                    index, timestamp, count, sorted(ids), missing)
        self.previous_ids = ids
        self.frames += 1

    def contacts(self, start, end, gap):
        sweep = start is not None and gap is not None and 0 < gap <= self.max_gap_frames
        result = {}
        for name in ("start", "finish", "final"):
            fraction = crossing_fraction(self.areas[name], start, end) if sweep else (
                1.0 if contains(self.areas[name], end) else None)
            if fraction is not None:
                result[name] = fraction
        return result

    @staticmethod
    def state(engine, track_id):
        state = engine.tracks.get(track_id)
        if state is None:
            return None
        return {"last_eligible_frame": state.last_frame,
                "last_eligible_centroid": list(state.last_centroid),
                "start_s": state.start_s, "finish_s": state.finish_s,
                "wrong_order": state.wrong_order, "event_written": state.event_written,
                "event_pending": state.event_pending,
                "violation_status": state.violation["violation_status"],
                "violation_reason": state.violation["violation_reason"]}

    def before(self, index, timestamp, detection, engine):
        point = centroid(detection.bbox_xyxy)
        prior = self.last_observed.get(detection.track_id) if detection.track_id is not None else None
        observed_gap = index - prior[0] if prior else None
        state = self.state(engine, detection.track_id)
        state_gap = index - state["last_eligible_frame"] if state else None
        record = {
            "type": "detection", "frame_index": index, "timestamp_s": timestamp,
            "track_id": detection.track_id, "category": detection.class_name,
            "vehicle_class_id": detection.class_id,
            "vehicle_class_source": detection.class_source,
            "confidence": detection.confidence, "bbox_xyxy": list(detection.bbox_xyxy),
            "centroid": list(point),
            "detector_bbox_xyxy": list(detection.detector_bbox_xyxy) if detection.detector_bbox_xyxy else None,
            "detector_confidence": detection.detector_confidence,
            "inside": {name: contains(area, point) for name, area in self.areas.items()},
            "previous_observed_frame": prior[0] if prior else None,
            "previous_observed_centroid": list(prior[1]) if prior else None,
            "observed_gap_frames": observed_gap,
            "observed_contacts": self.contacts(prior[1] if prior else None, point, observed_gap),
            "state_gap_frames": state_gap,
            "state_sweep_allowed": state is not None and 0 < state_gap <= self.max_gap_frames,
            "state_contacts_before_gate": self.contacts(
                tuple(state["last_eligible_centroid"]) if state else None, point, state_gap),
            "state_before": state,
        }
        if detection.track_id is not None:
            self.last_observed[detection.track_id] = (index, point)
        return record

    def after(self, record, engine, gate, event=False):
        record.update(gate=gate, event_written_this_frame=event,
                      state_after=self.state(engine, record["track_id"]))
        self.emit(record)
        LOGGER.info(
            "detection frame=%s t=%.6f id=%s bbox=%s centroid=%s inside=%s "
            "observed_contacts=%s state_gap=%s gate=%s state_after=%s",
            record["frame_index"], record["timestamp_s"], record["track_id"],
            record["bbox_xyxy"], record["centroid"], record["inside"],
            record["observed_contacts"], record["state_gap_frames"], gate, record["state_after"],
        )
        self.detections += 1

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()
