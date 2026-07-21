"""Pure validation helpers for the GPU flight test harness."""
from collections.abc import Sequence

import numpy as np


def sustained_recovery_time(
    errors: Sequence[float],
    start_index: int,
    *,
    threshold: float = 0.05,
    hold_steps: int = 20,
    dt: float = 0.01,
):
    """Return seconds until a sustained post-disturbance recovery.

    Recovery starts at the first sample in a run of ``hold_steps`` finite
    errors strictly below ``threshold``. ``None`` means the signal never
    demonstrated recovery and must not be interpreted as success.
    """
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if hold_steps <= 0:
        raise ValueError("hold_steps must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    values = np.asarray(errors, dtype=float)
    if start_index >= len(values):
        return None

    consecutive = 0
    for index in range(start_index, len(values)):
        value = values[index]
        if np.isfinite(value) and value < threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= hold_steps:
            recovery_start = index - hold_steps + 1
            return max(0, recovery_start - start_index) * dt
    return None


def disturbance_peak(errors: Sequence[float], start_index: int):
    """Return the largest finite error from disturbance onset onward."""
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    values = np.asarray(errors, dtype=float)
    if start_index >= len(values):
        return None
    values = values[start_index:]
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else None
