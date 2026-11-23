"""The NumPy version of the ``push_relativistic_boris`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``push_relativistic_boris_pyccel.py``, vectorized over the
markers (`mpiparticles.kernels.boris.relativistic_boris_update`).
"""

from __future__ import annotations

import numpy as np

from mpiparticles.kernels.boris import lorentz_factor, relativistic_boris_update


def push_relativistic_boris(
    positions: np.ndarray,
    proper_velocities: np.ndarray,
    e_field: np.ndarray,
    b_field: np.ndarray,
    charge_over_mass: float,
    dt: float,
    speed_of_light: float,
) -> None:
    """Advance the markers by one relativistic Boris step, in place (see the Pyccel version)."""
    half = 0.5 * charge_over_mass * dt
    inv_c2 = 1.0 / (speed_of_light * speed_of_light)
    proper_velocities[...] = relativistic_boris_update(
        proper_velocities, half * e_field, half * b_field, inv_c2
    )
    ndim = positions.shape[0]
    step = dt / lorentz_factor(proper_velocities, inv_c2)
    positions += step * proper_velocities[:ndim]
