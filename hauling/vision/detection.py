"""Local YOLO inference and a fresh ByteTrack instance for each run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import math

from hauling.config import PipelineConfig
from hauling.vision.plate import PlateCandidate, horizontal_hull_region, select_plate


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    track_id: int | None
    detector_bbox_xyxy: tuple[float, float, float, float] | None = None
    detector_confidence: float | None = None
    class_id: int | None = None
    class_name: str = "Vehicle"
    class_source: str = "detector"


def local_weights(path: Path) -> Path:
    if path.is_dir() and path.name.endswith("_openvino_model"):
        if (path / "metadata.yaml").is_file() and any(
            file.with_suffix(".bin").is_file() for file in path.glob("*.xml")
        ):
            return path.resolve()
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError(f"Model must be an existing local .pt file or complete OpenVINO directory: {path.name}")
    return path.resolve()


def validate_plate_classes(names: dict | list) -> None:
    labels = names.values() if isinstance(names, dict) else names
    labels = {str(name).lower().strip() for name in labels}
    if not labels:
        raise ValueError("Hull-region detector has no class labels")
    # The user confirmed No_Lambung's `Character` means the intended hull-number
    # region above the front windshield. Do not confuse it with OCR classifiers.


class YoloAdapter:
    def __init__(
        self, vehicle_path: Path, plate_path: Path | None, config: PipelineConfig,
        fps: float, device: str, tracker_path: Path, buffer_seconds: float,
        plate_confidence: float, threads: int,
        hull_min_aspect_ratio: float = 2.0,
        hull_reference: Path | None = None,
    ):
        vehicle_path = local_weights(vehicle_path)
        plate_path = local_weights(plate_path) if plate_path else None
        # Imports happen inside offline_runtime, never at package import time.
        import numpy as np
        import torch
        import yaml
        from ultralytics import YOLO, settings
        from ultralytics.trackers.byte_tracker import BYTETracker

        settings.update({key: False for key in (
            "sync", "hub", "wandb", "comet", "clearml", "mlflow", "neptune"
        ) if key in settings})

        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.device = "cuda:0" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu")
        for model_path in (vehicle_path, plate_path):
            if model_path is not None and model_path.is_dir():
                try:
                    with (model_path / "metadata.yaml").open(encoding="utf-8") as handle:
                        metadata = yaml.safe_load(handle)
                    if not isinstance(metadata, dict):
                        raise ValueError("Missing model metadata")
                except (OSError, ValueError, yaml.YAMLError):
                    raise ValueError("Cannot read local OpenVINO metadata; re-export the model") from None
                if metadata.get("task") != "detect":
                    raise ValueError("Exported model must have detection-task metadata")
                if self.device not in {"cpu", "intel:cpu", "intel:gpu", "intel:npu"}:
                    raise ValueError("Use cpu or intel:cpu/gpu/npu with exported OpenVINO models")
            elif model_path is not None and self.device.startswith("intel:"):
                raise ValueError("Intel devices require exported OpenVINO model directories")
        self.vehicle_device = "intel:cpu" if vehicle_path.is_dir() and self.device == "cpu" else self.device
        self.plate_device = "intel:cpu" if plate_path and plate_path.is_dir() and self.device == "cpu" else self.device
        try:
            self.vehicle = YOLO(str(vehicle_path), task="detect" if vehicle_path.is_dir() else None)
            self.plate = YOLO(str(plate_path), task="detect" if plate_path.is_dir() else None) if plate_path else None
        except Exception as error:
            raise ValueError(
                f"Cannot load local model ({type(error).__name__}); check weights "
                "and the documented PyTorch/Ultralytics versions"
            ) from None
        self.vehicle.overrides["device"] = self.vehicle_device
        if self.plate is not None:
            self.plate.overrides["device"] = self.plate_device
        if self.vehicle.task != "detect" or (self.plate is not None and self.plate.task != "detect"):
            raise ValueError("Vehicle and plate weights must be bounding-box detection models, not classifiers")
        if self.plate is not None:
            validate_plate_classes(self.plate.names)
        try:
            with tracker_path.open(encoding="utf-8") as handle:
                tracker_settings = yaml.safe_load(handle)
            if tracker_settings["tracker_type"] != "bytetrack":
                raise ValueError("Only local ByteTrack configurations are supported")
            tracker_settings["track_buffer"] = max(1, round(buffer_seconds * 30))
            self.tracker = BYTETracker(SimpleNamespace(**tracker_settings), frame_rate=fps)
        except Exception:
            raise ValueError("Cannot read ByteTrack config; use the bundled bytetrack.yaml") from None
        self.config = config
        self.plate_confidence = plate_confidence
        self.hull_min_aspect_ratio = hull_min_aspect_ratio
        self.reference_hull = None
        if hull_reference is not None:
            from hauling.vision.hull_reference import ReferenceHullDetector
            self.reference_hull = ReferenceHullDetector(hull_reference)
        self.hull_diagnostics = {
            "vehicle_crops_processed": 0,
            "raw_candidates_above_confidence": 0,
            "candidates_rejected_shape_or_bounds": 0,
            "accepted_candidates": 0,
            "crops_with_valid_region": 0,
        }
        self.reference_diagnostics = {"crops_processed": 0, "crops_with_panel": 0}

    @property
    def model_classes(self) -> dict:
        return {
            "vehicle": self.vehicle.names,
            "plate": self.plate.names if self.plate else None,
        }

    def detect(self, frame) -> tuple[list[Detection], int]:
        cfg = self.config
        result = self.vehicle.predict(
            frame, conf=cfg.confidence, iou=cfg.iou, imgsz=cfg.image_size,
            max_det=cfg.max_detections, agnostic_nms=cfg.agnostic_nms,
            device=self.vehicle_device, verbose=False, save=False, half=False,
        )[0]
        boxes = result.boxes.cpu().numpy()
        # Calling the tracker even with empty detections ages lost tracks.
        tracks = self.tracker.update(boxes, frame)
        assigned = {int(row[-1]): row for row in tracks}
        detections = []
        for index, (box, confidence, class_id) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
            row = assigned.get(index)
            detections.append(Detection(
                tuple(float(v) for v in (row[:4] if row is not None else box)),
                float(row[5]) if row is not None else float(confidence),
                int(row[4]) if row is not None else None,
                tuple(float(v) for v in box), float(confidence),
                int(class_id), str(self.vehicle.names[int(class_id)]),
            ))
        return detections, len(boxes)

    def detect_plate(self, frame, detection: Detection, timestamp_s: float) -> PlateCandidate | None:
        if self.plate is None:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox_xyxy
        left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
        right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if right <= left or bottom <= top:
            return None
        crop = frame[top:bottom, left:right]
        self.hull_diagnostics["vehicle_crops_processed"] += 1
        result = self.plate.predict(
            crop, conf=self.plate_confidence, iou=self.config.iou,
            imgsz=self.config.image_size, max_det=self.config.max_detections,
            device=self.plate_device, verbose=False, save=False, half=False,
        )[0]
        best = None
        for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            self.hull_diagnostics["raw_candidates_above_confidence"] += 1
            bx1, by1, bx2, by2 = box
            candidate = PlateCandidate(float(confidence), (
                max(left, min(right, math.floor(bx1 + left))),
                max(top, min(bottom, math.floor(by1 + top))),
                max(left, min(right, math.ceil(bx2 + left))),
                max(top, min(bottom, math.ceil(by2 + top))),
            ), timestamp_s)
            if horizontal_hull_region(candidate, self.hull_min_aspect_ratio):
                self.hull_diagnostics["accepted_candidates"] += 1
                best = select_plate(best, candidate)
            else:
                self.hull_diagnostics["candidates_rejected_shape_or_bounds"] += 1
        if best is not None:
            self.hull_diagnostics["crops_with_valid_region"] += 1
        elif getattr(self, "reference_hull", None) is not None:
            self.reference_diagnostics["crops_processed"] += 1
            best = self.reference_hull.detect(crop, (left, top), timestamp_s)
            if best is not None:
                self.reference_diagnostics["crops_with_panel"] += 1
        return best
