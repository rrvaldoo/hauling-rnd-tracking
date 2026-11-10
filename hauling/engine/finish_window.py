"""Bounded offline OCR windows anchored to immutable finish events."""

from dataclasses import dataclass

from hauling.engine.state import VehicleEvent


@dataclass(frozen=True)
class PendingFinish:
    event: VehicleEvent
    deadline_frame: int


class FinishWindows:
    def __init__(self, lookahead_frames: int):
        self.lookahead_frames = lookahead_frames
        self.pending: dict[int, PendingFinish] = {}

    def add(self, event: VehicleEvent) -> None:
        self.pending[event.track.track_id] = PendingFinish(event, event.frame_index + self.lookahead_frames)

    def contains(self, track_id: int) -> bool:
        return track_id in self.pending

    def ready(self, frame_index: int, force: bool = False) -> list[PendingFinish]:
        keys = [key for key, window in self.pending.items() if force or window.deadline_frame <= frame_index]
        return [self.pending.pop(key) for key in keys]
