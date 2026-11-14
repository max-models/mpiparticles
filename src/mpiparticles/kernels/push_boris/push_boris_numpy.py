"""The NumPy version of the ``push_boris`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``push_boris_pyccel.py``, vectorized over the markers.
"""

from __future__ import annotations

import numpy as np


def push_boris(
    positions: np.ndarray,
    velocities: np.ndarray,
    e_field: np.ndarray,
    b_field: np.ndarray,
    charge_over_mass: float,
    dt: float,
) -> None:
    """Advance the markers by one Boris step, in place (see the Pyccel version)."""
    half = 0.5 * charge_over_mass * dt
    v_minus = velocities + half * e_field
    t = half * b_field
    s = 2.0 * t / (1.0 + np.sum(t * t, axis=0))
    v_prime = v_minus + np.cross(v_minus, t, axis=0)
    velocities[...] = v_minus + np.cross(v_prime, s, axis=0) + half * e_field
    ndim = positions.shape[0]
    positions += dt * velocities[:ndim]
