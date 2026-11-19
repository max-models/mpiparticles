"""The NumPy version of the ``guiding_centre_rates`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``guiding_centre_rates_pyccel.py``, vectorized over the
markers (`mpiparticles.kernels.guiding_centre.guiding_centre_rates`).
"""

from __future__ import annotations

import numpy as np

from mpiparticles.kernels.guiding_centre import guiding_centre_rates as rates


def guiding_centre_rates(
    e_field: np.ndarray,
    b_field: np.ndarray,
    grad_b: np.ndarray,
    curvature: np.ndarray,
    v_par: np.ndarray,
    mu: np.ndarray,
    charge: float,
    mass: float,
    drift: np.ndarray,
    accel: np.ndarray,
) -> None:
    """Write ``dX/dt`` and ``dv_par/dt`` (see the Pyccel version)."""
    drift[...], accel[...] = rates(
        e_field, b_field, grad_b, curvature, v_par, mu, charge, mass
    )
