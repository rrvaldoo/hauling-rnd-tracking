"""Conservative camera-specific type refinement for the single-class detector."""

from dataclasses import replace
from pathlib import Path

from hauling.vision.detection import Detection, local_weights
from hauling.utils.geometry import centroid, contains, polygon_width_at_y


class VehicleTypeResolver:
    def __init__(self, path: Path, device: str, main_area, interval_frames: int,
                 light_max_width_ratio: float = .30):
        local_weights(path)
        if path.suffix.lower() != ".pt":
            raise ValueError("Vehicle type refinement requires a local .pt classifier")
        if not 0 < light_max_width_ratio < 1:
            raise ValueError("Light-vehicle width ratio must be between zero and one")
        from ultralytics import YOLO

        self.model = YOLO(str(path.resolve()))
        if self.model.task != "classify":
            raise ValueError("Vehicle type refinement requires a classification model")
        if set(self.model.names.values()) != {"truck batu bara", "truck non batu bara"}:
            raise ValueError("Vehicle type refinement expects the supplied coal/non-coal classifier")
        self.device = "cpu" if device.startswith("intel:") else device
        self.main_area = main_area
        self.interval_frames = max(1, interval_frames)
        self.light_max_width_ratio = light_max_width_ratio
        self.cache = {}
        self.last_inference = {}
        self.diagnostics = {"inferences": 0, "tracks": {}}

    def resolve(self, frame, detection: Detection, frame_index: int, frozen: bool = False) -> Detection:
        track_id = detection.track_id
        if track_id is None or detection.class_name not in {"Truck", "Light Vehicle"}:
            return detection
        previous = self.cache.get(track_id)
        point = centroid(detection.bbox_xyxy)
        due = frame_index - self.last_inference.get(track_id, -self.interval_frames) >= self.interval_frames
        if not frozen and due and contains(self.main_area, point):
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = detection.bbox_xyxy
            left, top = max(0, int(x1)), max(0, int(y1))
            right, bottom = min(width, int(x2)), min(height, int(y2))
            span = polygon_width_at_y(self.main_area, point[1])
            if right - left >= 64 and bottom - top >= 64 and span:
                result = self.model.predict(frame[top:bottom, left:right], imgsz=224,
                                            device=self.device, verbose=False, save=False, half=False)[0]
                name = self.model.names[int(result.probs.top1)]
                confidence = float(result.probs.top1conf)
                ratio = (x2 - x1) / span
                # 'Non-coal' alone is NOT a Light Vehicle label. A small footprint
                # relative to this camera's calibrated ROI is also required.
                if confidence >= .9:
                    if name == "truck batu bara":
                        previous = (0, "Truck")
                    elif ratio <= self.light_max_width_ratio:
                        previous = (1, "Light Vehicle")
                    else:
                        previous = None
                else:
                    previous = None
                self.cache[track_id] = previous
                self.last_inference[track_id] = frame_index
                self.diagnostics["inferences"] += 1
                self.diagnostics["tracks"][str(track_id)] = {
                    "frame_index": frame_index, "secondary_label": name,
                    "secondary_confidence": confidence, "roi_width_ratio": ratio,
                    "resolved_class": previous[1] if previous else detection.class_name,
                }
        if previous:
            return replace(detection, class_id=previous[0], class_name=previous[1],
                           class_source="coal_classifier_and_camera_width")
        return detection
