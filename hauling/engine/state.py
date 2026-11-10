"""The per-track source of truth; independent of model and video libraries."""

from dataclasses import dataclass, field
import math
from decimal import Decimal

from hauling.config import PipelineConfig, Point, Polygon
from hauling.utils.geometry import contains, crossing_fraction
from hauling.vision.plate import PlateCandidate
from hauling.engine.speed import SpeedMeasurement, measure_speed
from hauling.engine.violation import evaluate_overspeed


@dataclass
class TrackState:
    track_id: int
    first_centroid: Point
    last_centroid: Point
    vehicle_confidence: float
    last_frame: int
    start_s: float | None = None
    finish_s: float | None = None
    wrong_order: bool = False
    plate: PlateCandidate | None = None
    event_written: bool = False
    vehicle_class_id: int | None = None
    vehicle_class: str = "Vehicle"
    vehicle_class_source: str = "detector"
    ocr_window_truncated: bool = False
    event_pending: bool = False
    frozen_measurement: SpeedMeasurement | None = None
    violation: dict = field(default_factory=lambda: evaluate_overspeed(False, None, None))

    @property
    def direction(self) -> str:
        delta = self.last_centroid[1] - self.first_centroid[1]
        if abs(delta) < 1e-6:
            return "Unknown"
        return "South" if delta > 0 else "North"

    def measurement(self, config: PipelineConfig) -> SpeedMeasurement:
        if self.frozen_measurement is not None:
            return self.frozen_measurement
        return measure_speed(self.start_s, self.finish_s, config.distance_meter,
                             config.speed_limit_kmh, self.wrong_order)


@dataclass(frozen=True)
class VehicleEvent:
    track: TrackState
    timestamp_s: float
    frame_index: int
    measurement: SpeedMeasurement


class TrackEngine:
    def __init__(self, config: PipelineConfig, areas: dict[str, Polygon], max_crossing_gap_frames: int = 1):
        if not math.isfinite(config.speed_limit_kmh) or config.speed_limit_kmh <= 0:
            raise ValueError("Speed limit must be positive and finite")
        self.config = config
        self.areas = areas
        self.tracks: dict[int, TrackState] = {}
        self.max_crossing_gap_frames = max_crossing_gap_frames

    def observe(
        self, track_id: int | None, point: Point, confidence: float,
        timestamp_s: float, frame_index: int, plate: PlateCandidate | None = None,
        vehicle_class_id: int | None = None, vehicle_class: str = "Vehicle",
        vehicle_class_source: str = "detector",
    ) -> VehicleEvent | None:
        if track_id is None or (track_id not in self.tracks and not contains(self.areas["main"], point)):
            return None
        state = self.tracks.get(track_id)
        if state is None:
            state = TrackState(track_id, point, point, confidence, frame_index)
            self.tracks[track_id] = state
        if state.event_written or state.event_pending:
            return None
        previous_point = state.last_centroid
        can_sweep = 0 < frame_index - state.last_frame <= self.max_crossing_gap_frames
        state.last_centroid = point
        state.last_frame = frame_index
        state.vehicle_confidence = confidence
        state.vehicle_class_id = vehicle_class_id
        state.vehicle_class = vehicle_class
        state.vehicle_class_source = vehicle_class_source
        contacts = []
        for order, name in enumerate(("start", "finish", "final")):
            fraction = crossing_fraction(self.areas[name], previous_point, point) if can_sweep else (
                1.0 if contains(self.areas[name], point) else None
            )
            if fraction is not None:
                contacts.append((fraction, order, name))
        # Segment order matters for reverse travel or multiple bands crossed in
        # one frame. Timestamps remain the current observed video's frame time.
        for _, _, name in sorted(contact for contact in contacts if contact[2] != "final"):
            if name == "start" and state.start_s is None:
                state.start_s = timestamp_s
            elif name == "finish" and state.finish_s is None:
                state.finish_s = timestamp_s
                if state.start_s is None:
                    state.wrong_order = True
                state.plate = plate if plate is not None and plate.valid and plate.timestamp_s == timestamp_s else None
                state.frozen_measurement = state.measurement(self.config)
                speed = state.frozen_measurement
                # measure_speed compares Decimal values before display rounding.
                state.violation = evaluate_overspeed(
                    speed.valid, speed.unrounded_speed, Decimal(str(self.config.speed_limit_kmh)))
        # Finish is evaluated first even when final is contacted in this frame.
        if any(name == "final" for _, _, name in contacts):
            state.event_pending = True
            return VehicleEvent(state, timestamp_s, frame_index, state.measurement(self.config))
        return None
