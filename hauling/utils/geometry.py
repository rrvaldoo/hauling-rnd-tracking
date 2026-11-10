"""Dependency-free, boundary-inclusive centroid geometry."""

from hauling.config import Point, Polygon


def scale_polygon(polygon: Polygon, base: tuple[int, int], target: tuple[int, int]) -> Polygon:
    if min(*base, *target) <= 0:
        raise ValueError("Resolutions must be positive")
    return tuple((x * target[0] / base[0], y * target[1] / base[1]) for x, y in polygon)


def contains(polygon: Polygon, point: Point) -> bool:
    x, y = point
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) <= 1e-7 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def centroid(box: tuple[float, float, float, float]) -> Point:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def polygon_width_at_y(polygon: Polygon, y: float) -> float | None:
    """Horizontal camera-relative reference span; independent of frame resolution."""
    intersections = []
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if ay == by:
            if y == ay:
                intersections.extend((ax, bx))
        elif min(ay, by) <= y <= max(ay, by):
            intersections.append(ax + (y - ay) * (bx - ax) / (by - ay))
    width = max(intersections) - min(intersections) if intersections else 0
    return width if width > 0 else None


def crossing_fraction(polygon: Polygon, start: Point, end: Point) -> float | None:
    """First boundary-inclusive contact along an observed centroid segment."""
    if contains(polygon, start):
        return 0.0
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return None
    candidates = [1.0] if contains(polygon, end) else []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        ex, ey = b[0] - a[0], b[1] - a[1]
        qx, qy = a[0] - start[0], a[1] - start[1]
        denominator = dx * ey - dy * ex
        if abs(denominator) > 1e-9:
            t = (qx * ey - qy * ex) / denominator
            u = (qx * dy - qy * dx) / denominator
            if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
                candidates.append(max(0.0, min(1.0, t)))
        elif abs(qx * dy - qy * dx) < 1e-9:
            # Collinear contact with an edge, including its endpoints.
            projections = [((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / length_squared for p in (a, b)]
            lo, hi = sorted(projections)
            if hi >= 0 and lo <= 1:
                candidates.append(max(0.0, lo))
    return min(candidates) if candidates else None
