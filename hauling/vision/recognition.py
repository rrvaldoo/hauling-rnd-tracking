"""Local two-stage hull OCR: locate digits/letters, classify, then order them."""

from dataclasses import dataclass, replace
from itertools import combinations
import math
from pathlib import Path

from hauling.vision.detection import local_weights
from hauling.vision.plate import HullCharacter, PlateCandidate


@dataclass(frozen=True)
class CharacterModels:
    digit_detector: Path
    letter_detector: Path
    digit_classifier: Path
    letter_classifier: Path

    def validate(self) -> None:
        for path in vars(self).values():
            local_weights(path)
            if path.suffix.lower() != ".pt":
                raise ValueError("Character recognition currently requires local .pt weights")


def overlap_fraction(first: HullCharacter, second: HullCharacter) -> float:
    """IoU suppresses duplicate proposals without swallowing adjacent narrow I/1."""
    a, b = first.bbox_xyxy, second.bbox_xyxy
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def ordered_characters(candidates: list[HullCharacter]) -> tuple[HullCharacter, ...]:
    selected = []
    for candidate in sorted(candidates, key=lambda c: (-c.confidence, c.bbox_xyxy, c.text)):
        if all(overlap_fraction(candidate, previous) < 0.45 for previous in selected):
            selected.append(candidate)
    return tuple(sorted(selected, key=lambda c: (c.bbox_xyxy[0] + c.bbox_xyxy[2], c.bbox_xyxy[1])))


@dataclass(frozen=True)
class GlyphReading:
    proposal: HullCharacter
    letter: tuple[str, float]
    digit: tuple[str, float]


def decode_hull(readings: list[GlyphReading], confidence: float, letters: int = 3,
                digits: int = 4) -> tuple[tuple[HullCharacter, ...], str]:
    """Choose an observed glyph sequence under the confirmed fleet format.

    The format selects classifier alphabets, never particular identifiers.
    Uncertain observed glyphs are '?'; missing boxes are never manufactured.
    """
    count = letters + digits
    if not readings:
        return (), "unread"
    # Bound the combinatorial search and retain spatial order.
    readings = sorted(sorted(readings, key=lambda r: -r.proposal.confidence)[:16],
                      key=lambda r: sum(r.proposal.bbox_xyxy[::2]))
    best, best_score = None, -math.inf
    for sequence in combinations(readings, min(count, len(readings))):
        widths = sorted(r.proposal.bbox_xyxy[2] - r.proposal.bbox_xyxy[0] for r in sequence)
        typical_width = widths[len(widths) // 2]
        score, valid = 0.0, True
        for index, reading in enumerate(sequence):
            _, probability = reading.letter if index < letters else reading.digit
            score += math.log(max(.001, probability * reading.proposal.confidence))
            if index:
                previous = sequence[index - 1].proposal.bbox_xyxy
                box = reading.proposal.bbox_xyxy
                step = (box[0] + box[2] - previous[0] - previous[2]) / 2
                if not .35 * typical_width <= step <= 2.5 * typical_width:
                    valid = False
                    break
                # Penalize skipped internal glyphs, but allow the prefix/suffix space.
                expected_step = 1.3 if index == letters else .9
                score -= .7 * abs(step / typical_width - expected_step)
        if valid and score > best_score:
            best, best_score = sequence, score
    if best is None:
        return (), "unread"
    characters = []
    for index, reading in enumerate(best):
        text, probability = reading.letter if index < letters else reading.digit
        characters.append(HullCharacter(text if probability >= confidence else "?",
                                        probability * reading.proposal.confidence,
                                        reading.proposal.bbox_xyxy))
    complete = len(characters) == count and all(c.text != "?" for c in characters)
    return tuple(characters), "complete" if complete else "partial"


class HullRecognizer:
    def __init__(self, paths: CharacterModels, device: str, confidence: float = 0.5):
        paths.validate()
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("Character confidence must be in (0, 1]")
        from ultralytics import YOLO

        # The supplied OCR weights are PyTorch models even when detection uses OpenVINO.
        self.device = "cpu" if device.startswith("intel:") else device
        self.confidence = confidence
        self.models = {}
        try:
            for name, path in vars(paths).items():
                model = YOLO(str(path.resolve()))
                expected = "detect" if name.endswith("detector") else "classify"
                if model.task != expected:
                    raise ValueError(f"{name} requires a {expected} model")
                self.models[name] = model
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"Cannot load local character models ({type(error).__name__})") from None
        for kind, allowed in (("digit", "0123456789"), ("letter", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
            names = self.models[f"{kind}_classifier"].names
            labels = names.values() if isinstance(names, dict) else names
            if not all(str(label) in allowed and len(str(label)) == 1 for label in labels):
                raise ValueError(f"Unexpected labels in {kind} classifier")

    @property
    def model_classes(self) -> dict:
        return {name: model.names for name, model in self.models.items()}

    def _classify_many(self, images: list, kind: str, contexts: list | None = None) -> list[tuple[str, float]]:
        if not images:
            return []
        classifier = self.models[f"{kind}_classifier"]

        def predict(views):
            results = classifier.predict(views, batch=min(16, len(views)), imgsz=96,
                                         device=self.device, verbose=False, save=False, half=False)
            readings = []
            for result in results:
                probability = float(result.probs.top1conf)
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    readings.append(("?", 0.0))
                else:
                    readings.append((str(classifier.names[int(result.probs.top1)]), probability))
            if len(readings) != len(views):
                raise ValueError("Character classifier returned an unexpected result count")
            return readings

        original = predict(images)
        indices = [i for i, (image, reading) in enumerate(zip(images, original))
                   if reading[1] < .95 and image.shape[1] >= 7]
        if indices:
            insets = predict([images[i][:, 1:-1] for i in indices])
            for index, inset in zip(indices, insets):
                if inset[1] > original[index][1] + .15:
                    original[index] = inset
        if contexts is not None:
            indices = [i for i, reading in enumerate(original)
                       if reading[1] < .8 and contexts[i].shape[0] > images[i].shape[0]]
            if indices:
                expanded = predict([contexts[i] for i in indices])
                for index, reading in zip(indices, expanded):
                    if reading[1] > original[index][1] + .15:
                        original[index] = reading
        return original

    def recognize(self, frame, plate: PlateCandidate) -> PlateCandidate:
        import numpy as np

        x1, y1, x2, y2 = plate.bbox_xyxy
        height, width = frame.shape[:2]
        left, top, right, bottom = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if not plate.valid or right <= left or bottom <= top:
            return replace(plate, characters=(), text_status="unread")
        crop = frame[top:bottom, left:right]
        blue, green, red = (crop[:, :, i].astype(np.float32) for i in range(3))
        ink = (red - green > 8) & (red - blue > 4) & (red > 80)
        # Apply the red-ink gate only when the panel is visibly red. Glared
        # suffix digits can lose all chroma and must still reach the classifiers.
        use_ink = float(ink.mean()) >= .15 and plate.source != "reference_glare_extension"
        proposals = []
        for kind in ("digit", "letter"):
            detector = self.models[f"{kind}_detector"]
            result = detector.predict(crop, conf=0.25, imgsz=640, max_det=100,
                                      device=self.device, verbose=False, save=False, half=False)[0]
            for box, detection_confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                if not all(math.isfinite(value) for value in (*box, detection_confidence)):
                    continue
                bx1, by1, bx2, by2 = box
                cx1, cy1 = max(0, math.floor(bx1)), max(0, math.floor(by1))
                cx2, cy2 = min(right - left, math.ceil(bx2)), min(bottom - top, math.ceil(by2))
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                if use_ink and float(ink[cy1:cy2, cx1:cx2].mean()) < .08:
                    continue
                proposals.append(HullCharacter("?", float(detection_confidence),
                                                (cx1, cy1, cx2, cy2)))
        proposals = list(ordered_characters(proposals))
        if proposals:
            typical_height = float(np.median([p.bbox_xyxy[3] - p.bbox_xyxy[1] for p in proposals]))
            proposals = [p for p in proposals if p.bbox_xyxy[3] - p.bbox_xyxy[1] >= typical_height * .55]
        if len(proposals) >= 4:
            centers = [(p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2 for p in proposals]
            bottoms = [p.bbox_xyxy[3] for p in proposals]
            slopes = [(bottoms[j] - bottoms[i]) / (centers[j] - centers[i])
                      for i in range(len(proposals)) for j in range(i + 1, len(proposals))
                      if centers[j] - centers[i] >= typical_height]
            slope = float(np.median(slopes)) if slopes else 0.0
            intercept = float(np.median([y - slope * x for x, y in zip(centers, bottoms)]))
            proposals = [p for p, x, y in zip(proposals, centers, bottoms)
                         if abs(y - slope * x - intercept) <= typical_height * .5]
        images, contexts, translated_proposals = [], [], []
        for proposal in proposals:
            cx1, cy1, cx2, cy2 = proposal.bbox_xyxy
            images.append(crop[cy1:cy2, cx1:cx2])
            contexts.append(crop[max(0, cy1 - 2):cy2, cx1:cx2])
            translated_proposals.append(replace(proposal, bbox_xyxy=(cx1 + left, cy1 + top, cx2 + left, cy2 + top)))
        letters = self._classify_many(images, "letter", contexts)
        digits = self._classify_many(images, "digit", contexts)
        readings = [GlyphReading(proposal, letter, digit)
                    for proposal, letter, digit in zip(translated_proposals, letters, digits)]
        characters, status = decode_hull(readings, self.confidence,
                                         getattr(self, "prefix_letters", 3), getattr(self, "suffix_digits", 4))
        return replace(plate, characters=characters, text_status=status)
