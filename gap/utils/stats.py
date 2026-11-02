"""Small statistics helpers. OWNER: done"""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (low, high) in [0, 1]."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside [0, {n}]")
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fmt_rate(successes: int, n: int) -> str:
    """'0.83 [0.75, 0.89] (83/100)'"""
    lo, hi = wilson_interval(successes, n)
    return f"{successes / n:.2f} [{lo:.2f}, {hi:.2f}] ({successes}/{n})" if n else "n/a"
