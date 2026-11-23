"""The NumPy version of the ``relativistic_boris_velocity`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``relativistic_boris_velocity_pyccel.py``, vectorized over the
markers (`mpiparticles.kernels.boris.relativistic_boris_update`).
"""

from __future__ import annotations

import numpy as np

from mpiparticles.kernels.boris import relativistic_boris_update


def relativistic_boris_velocity(
    proper_velocities: np.ndarray,
    e_field: np.ndarray,
    b_field: np.ndarray,
    charge_over_mass: np.ndarray,
    dt: np.ndarray,
    speed_of_light: float,
) -> None:
    """Advance the proper velocities by one relativistic Boris step, in place."""
    half = 0.5 * charge_over_mass * dt
    proper_velocities[...] = relativistic_boris_update(
        proper_velocities,
        half * e_field,
        half * b_field,
        1.0 / (speed_of_light * speed_of_light),
    )
