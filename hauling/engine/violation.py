"""Pure overspeed evaluation as specified for KM-18."""


def evaluate_overspeed(measurement_valid, speed_kmh, speed_limit_kmh):
    if not measurement_valid:
        return {"is_overspeed": False, "violation_status": "NOT_EVALUATED",
                "violation_reason": None}
    if speed_kmh > speed_limit_kmh:
        return {"is_overspeed": True, "violation_status": "OVERSPEED",
                "violation_reason": "Speed exceeds configured limit"}
    return {"is_overspeed": False, "violation_status": "NORMAL",
            "violation_reason": None}
