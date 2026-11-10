"""Hull-region candidates in full-frame coordinates; PRD plate_* compatibility."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HullCharacter:
    text: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class HullObservation:
    timestamp_s: float
    text: str
    text_status: str
    confidence: float | None


@dataclass(frozen=True)
class PlateCandidate:
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    timestamp_s: float
    source: str = "yolo"
    characters: tuple[HullCharacter, ...] = ()
    text_status: str = "not_run"
    observations: tuple[HullObservation, ...] = ()

    @property
    def text(self) -> str:
        return "".join(character.text for character in self.characters)

    @property
    def text_confidence(self) -> float | None:
        return min((character.confidence for character in self.characters), default=None)

    @property
    def valid(self) -> bool:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (
            math.isfinite(self.confidence) and 0 <= self.confidence <= 1
            and math.isfinite(self.timestamp_s) and self.timestamp_s >= 0
            and x2 > x1 >= 0 and y2 > y1 >= 0
        )


def select_plate(best: PlateCandidate | None, candidate: PlateCandidate | None) -> PlateCandidate | None:
    if candidate is None or not candidate.valid:
        return best
    if best is None or (candidate.confidence, candidate.timestamp_s) > (best.confidence, best.timestamp_s):
        return candidate
    return best


def horizontal_hull_region(candidate: PlateCandidate, minimum_aspect_ratio: float = 2.0) -> bool:
    """Reject narrow vertical false positives shown beside the user's target.

    The red-circled target in contoh-hull is a horizontal number strip. This
    filter cannot manufacture a missing region or establish model accuracy.
    """
    if not candidate.valid:
        return False
    x1, y1, x2, y2 = candidate.bbox_xyxy
    return (x2 - x1) / (y2 - y1) >= minimum_aspect_ratio
