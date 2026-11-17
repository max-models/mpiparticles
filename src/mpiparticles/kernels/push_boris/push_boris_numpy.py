"""The NumPy version of the ``push_boris`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``push_boris_pyccel.py``, vectorized over the markers (`mpiparticles.kernels.boris.boris_update`).
"""

from __future__ import annotations

import numpy as np

from mpiparticles.kernels.boris import boris_update


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
    velocities[...] = boris_update(velocities, half * e_field, half * b_field)
    ndim = positions.shape[0]
    positions += dt * velocities[:ndim]
