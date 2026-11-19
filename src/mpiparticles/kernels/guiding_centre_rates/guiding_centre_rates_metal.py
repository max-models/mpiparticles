"""The Metal version (Apple-silicon GPU, float32) of the ``guiding_centre_rates`` kernel.

``guiding_centre_rates_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker. Import only where `cunumpy.kernels.metal_available` is
true; see `mpiparticles.kernels`.
"""

from __future__ import annotations

import numpy as np

from mpiparticles._types import Array
from mpiparticles.kernels import KERNEL_PACKAGE
from mpiparticles.kernels.guiding_centre import GUIDING_CENTRE_METAL_HEADER

_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint n = v_par_shape[0];
    if (k >= n) return;
    float3 e = float3(e_field[k], e_field[n + k], e_field[2 * n + k]);
    float3 b = float3(b_field[k], b_field[n + k], b_field[2 * n + k]);
    float3 g = float3(grad_b[k], grad_b[n + k], grad_b[2 * n + k]);
    float3 c = float3(curvature[k], curvature[n + k], curvature[2 * n + k]);
    float4 r = mpiparticles_gc_rates(e, b, g, c, v_par[k], mu[k], species[0], species[1]);
    drift[k] = r.x;
    drift[n + k] = r.y;
    drift[2 * n + k] = r.z;
    accel[k] = r.w;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "mpiparticles_guiding_centre_rates",
    _SOURCE,
    inputs=["e_field", "b_field", "grad_b", "curvature", "v_par", "mu", "species"],
    outputs=["drift", "accel"],
    header=GUIDING_CENTRE_METAL_HEADER,
)


def guiding_centre_rates(
    e_field: Array,
    b_field: Array,
    grad_b: Array,
    curvature: Array,
    v_par: Array,
    mu: Array,
    charge: float,
    mass: float,
    drift: Array,
    accel: Array,
) -> None:
    """Write ``dX/dt`` and ``dv_par/dt`` on the GPU (see the Pyccel version)."""
    n = v_par.shape[0]
    if not n:
        return
    new_drift = np.zeros((3, n), dtype=np.float32)
    new_accel = np.zeros(n, dtype=np.float32)
    _KERNEL(
        e_field,
        b_field,
        grad_b,
        curvature,
        v_par,
        mu,
        np.array([charge, mass]),
        out=(new_drift, new_accel),
        n_threads=n,
    )
    drift[...] = new_drift
    accel[...] = new_accel
