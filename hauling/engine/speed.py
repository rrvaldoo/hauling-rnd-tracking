from dataclasses import dataclass
from decimal import Decimal
import math

from hauling.engine.violation import evaluate_overspeed


@dataclass(frozen=True)
class SpeedMeasurement:
    status: str
    speed_kmh: float | None = None
    elapsed_seconds: float | None = None
    is_overspeed: bool = False
    unrounded_speed: Decimal | None = None

    @property
    def valid(self) -> bool:
        return self.status == "valid"


def measure_speed(
    start: float | None, finish: float | None,
    distance: float, speed_limit: float, wrong_order: bool = False,
) -> SpeedMeasurement:
    if wrong_order:
        return SpeedMeasurement("wrong_order")
    if start is None:
        return SpeedMeasurement("not_crossed_start")
    if finish is None:
        return SpeedMeasurement("not_crossed_finish")
    if not math.isfinite(start) or not math.isfinite(finish):
        return SpeedMeasurement("invalid_elapsed_time")
    elapsed = Decimal(str(finish)) - Decimal(str(start))
    if elapsed <= 0:
        return SpeedMeasurement("invalid_elapsed_time")
    # Decimal arithmetic avoids falsely flagging an exact limit because binary
    # floating point evaluates e.g. 60 / 3.6 * 3.6 as 60.00000000000001.
    speed = Decimal(str(distance)) * Decimal("3.6") / elapsed
    if not math.isfinite(float(speed)):
        return SpeedMeasurement("invalid_speed")
    evaluation = evaluate_overspeed(True, speed, Decimal(str(speed_limit)))
    return SpeedMeasurement("valid", float(speed), float(elapsed), evaluation["is_overspeed"], speed)
