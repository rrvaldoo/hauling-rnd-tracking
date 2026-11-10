"""Reference appearance fallback for red-on-white windshield hull panels.

This is a local color/context detector, not a trained YOLO model or OCR. It
requires observed panel-colored pixels, a light backing, and glass below the
panel. Relative search limits reject roof beacons and lower license plates.
"""

from pathlib import Path

from hauling.vision.plate import PlateCandidate


class ReferenceHullDetector:
    def __init__(self, reference_path: Path):
        import cv2
        import numpy as np

        reference = cv2.imread(str(reference_path))
        if reference is None:
            raise ValueError("Cannot read the hull-panel reference image")
        hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        red = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 65)
        white = (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 125)
        ys, xs = np.nonzero(red)
        if len(xs) < 10 or white.mean() < .1:
            raise ValueError("Reference must show a red-on-white hull-number panel")
        self.reference_aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
        self.reference_ink_width = int(xs.max() - xs.min() + 1)
        self.reference_path = reference_path

    def detect(self, crop, origin: tuple[int, int], timestamp_s: float):
        import cv2
        import numpy as np

        h, w = crop.shape[:2]
        if h < 30 or w < 30:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        blue, green, red_channel = cv2.split(crop.astype(np.float32))
        strong = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165))
                  & (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 75))
        # Weak red survives the pickup's glare; it still requires red dominance.
        weak = (red_channel - green > 4) & (red_channel - blue > 1) & (red_channel > 80)
        proposals = []
        glare_proposals = []
        for mask, kernel in (
            (strong, (max(3, round(w * .02)), 3)),
            (weak, (max(7, round(w * .12)), max(3, round(h * .008)))),
        ):
            binary = mask.astype(np.uint8) * 255
            grouped = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
            contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                gx, gy, gw, gh = cv2.boundingRect(contour)
                ys, xs = np.nonzero(binary[gy:gy + gh, gx:gx + gw])
                if len(xs) < 8:
                    continue
                x, y = gx + int(xs.min()), gy + int(ys.min())
                bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
                aspect = bw / bh
                if not (max(10, w * .05) <= bw <= w * .6 and 3 <= bh <= h * .08
                        and 1.5 <= aspect <= 12
                        and .18 <= (y + bh / 2) / h <= .76):
                    continue
                # Separate ink blobs distinguish a printed panel from a solid
                # red cargo rail. This checks texture without reading letters.
                _, _, components, _ = cv2.connectedComponentsWithStats(binary[y:y + bh, x:x + bw])
                ink_blobs = sum(
                    area >= 2 and ch >= max(2, bh * .2) and cw < bw * .6
                    for _, _, cw, ch, area in components[1:]
                )
                if ink_blobs < 2:
                    continue
                # Include the backing and a small border, not just red ink.
                px, py = max(2, round(bh * .55)), max(1, round(bh * .25))
                x1, y1 = max(0, x - px), max(0, y - py)
                x2, y2 = min(w, x + bw + px), min(h, y + bh + py)
                patch = hsv[y1:y2, x1:x2]
                white = float(np.mean((patch[:, :, 1] < 90) & (patch[:, :, 2] > 125)))
                ink = float(mask[y1:y2, x1:x2].mean())
                bottom = min(h, y2 + max(bh * 3, 6))
                below = hsv[y2:bottom, x1:x2]
                below_crop = crop[y2:bottom, x1:x2].astype(np.float32)
                if below.size == 0:
                    continue
                glass = ((below[:, :, 2] < 150)
                         | ((below_crop[:, :, 0] + below_crop[:, :, 1]) / 2
                            - below_crop[:, :, 2] > 8))
                context = float(glass.mean())
                if white < .18 or not .06 <= ink <= .65 or context < .55:
                    continue
                full_panel = bw >= max(12, w * .08) and aspect >= max(2.5, self.reference_aspect * .5)
                if not full_panel:
                    # Glare can leave only a short red prefix. Recover the light
                    # backing beside it, while still requiring observed ink and
                    # windshield context. Do not invent missing character text.
                    if white < .6:
                        continue
                    ex1 = max(0, x - round(bh * .8))
                    ey1 = max(0, y - bh)
                    ex2 = min(w, x + round(max(bw * 3.5, bh * self.reference_aspect * 1.8)))
                    ey2 = min(h, y + round(bh * 3))
                    backing = hsv[ey1:ey2, ex1:ex2]
                    backing_fraction = float(np.mean((backing[:, :, 1] < 100) & (backing[:, :, 2] > 125)))
                    if backing_fraction < .5:
                        continue
                    ox, oy = origin
                    glare_proposals.append(PlateCandidate(
                        float(.5 * backing_fraction + .3 * context),
                        (ox + ex1, oy + ey1, ox + ex2, oy + ey2), timestamp_s,
                        source="reference_glare_extension",
                    ))
                    continue
                shape = min(aspect, self.reference_aspect) / max(aspect, self.reference_aspect)
                # A bounded appearance score; explicitly not YOLO confidence.
                score = .35 * min(1., white / .5) + .35 * context + .2 * shape + .1 * min(1., ink / .25)
                # A few crisp prefix letters can resemble the reference while
                # glare hides the rest. Prefer fuller, better sampled panels
                # across frames instead of letting that tiny fragment win.
                score *= min(1., bw / self.reference_ink_width)
                ox, oy = origin
                proposals.append(PlateCandidate(
                    float(score), (ox + x1, oy + y1, ox + x2, oy + y2), timestamp_s,
                    source="reference_appearance",
                ))
        return max(proposals or glare_proposals, key=lambda item: item.confidence, default=None)
