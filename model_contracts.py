"""
Model contracts and validation helpers.

These are intentionally lightweight; wire them into hot paths only when
debugging to avoid perturbing performance.
"""
from __future__ import annotations

import numpy as np


def assert_finite(arr, name: str = "array"):
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


def assert_range(x, lo: float, hi: float, name: str = "value"):
    if x < lo or x > hi:
        raise ValueError(f"{name} out of range [{lo}, {hi}]: {x}")


def assert_shape(arr, shape, name: str = "array"):
    if arr.shape != shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
