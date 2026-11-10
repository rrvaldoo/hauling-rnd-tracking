"""Read an allowlist of inference settings without using production settings."""

from __future__ import annotations

import configparser
import json
import math
from dataclasses import dataclass
from pathlib import Path

Point = tuple[float, float]
Polygon = tuple[Point, ...]
AREA_KEYS = {
    "main": "main_area_coordinates_list",
    "start": "speed_area_start_coordinates_list",
    "finish": "speed_area_final_coordinates_list",
    "final": "final_area_coordinates_list",
}


@dataclass(frozen=True)
class PipelineConfig:
    resolution: tuple[int, int]
    areas: dict[str, Polygon]
    distance_meter: float
    speed_limit_kmh: float
    travel_direction: str
    confidence: float = 0.25
    iou: float = 0.7
    image_size: int = 640
    max_detections: int = 100
    agnostic_nms: bool = True


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Expected a finite number")
    return result


def parse_polygon(value: str) -> Polygon:
    points = json.loads(value)
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Polygon requires at least three points")
    result = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Each polygon point requires x and y")
        result.append((_number(point[0]), _number(point[1])))
    area = sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(result, result[1:] + result[:1])
    )
    if abs(area) < 1e-9:
        raise ValueError("Polygon must have nonzero area")
    return tuple(result)


def read_config(path: Path) -> PipelineConfig:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8-sig") as handle:
            parser.read_file(handle)
        section = parser["CONFIG"]
        width, height = (int(x.strip()) for x in section["resolution_area"].split(","))
        if width <= 0 or height <= 0:
            raise ValueError("Resolution must be positive")
        areas = {name: parse_polygon(section[key]) for name, key in AREA_KEYS.items()}
        distance = json.loads(section["distance_area_speed"])
        if isinstance(distance, list):
            if len(distance) != 1:
                raise ValueError("Exactly one speed distance is required")
            distance = distance[0]
        distance = _number(distance)
        limit = _number(json.loads(section["speed_limit"]))
        confidence = section.getfloat("conf_model", 0.25)
        iou = section.getfloat("iou_model", 0.7)
        image_size = section.getint("imgsz_model", 640)
        max_detections = section.getint("max_det_model", 100)
        direction = section["travel_direction"].strip()
        if distance <= 0 or limit <= 0:
            raise ValueError("Distance and speed limit must be positive")
        if not 0 < confidence <= 1 or not 0 < iou <= 1:
            raise ValueError("Confidence and IoU must be in (0, 1]")
        if image_size <= 0 or max_detections <= 0:
            raise ValueError("Image size and detection limit must be positive")
        if direction not in {"South", "North", "Unknown"}:
            raise ValueError("Unsupported travel direction")
        return PipelineConfig(
            (width, height), areas, distance, limit, direction,
            confidence, iou, image_size, max_detections,
            section.getboolean("agnostic_nms_model", True),
        )
    except (OSError, ValueError, KeyError, configparser.Error):
        # Do not include raw parser exceptions: they may contain secrets from
        # unrelated production sections in the supplied configuration.
        raise ValueError(
            "Cannot read KM-18 config: check [CONFIG], required area/distance/"
            "speed/direction keys, JSON polygons, and numeric inference settings"
        ) from None
