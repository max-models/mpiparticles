"""The NumPy version of the ``guiding_centre_velocity`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``guiding_centre_velocity_pyccel.py``, vectorized over the
markers (`mpiparticles.kernels.guiding_centre.guiding_centre_velocity`).
"""

from __future__ import annotations

import numpy as np

from mpiparticles.kernels.guiding_centre import guiding_centre_velocity as velocity


def guiding_centre_velocity(
    velocities: np.ndarray,
    b_old: np.ndarray,
    b_new: np.ndarray,
    v_par: np.ndarray,
    mu: np.ndarray,
    mass: float,
) -> None:
    """Replace the velocities in place (see the Pyccel version)."""
    velocities[...] = velocity(velocities, b_old, b_new, v_par, mu, mass)
