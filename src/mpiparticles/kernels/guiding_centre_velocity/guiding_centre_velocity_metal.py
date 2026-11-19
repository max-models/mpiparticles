"""The Metal version (Apple-silicon GPU, float32) of the ``guiding_centre_velocity`` kernel.

``guiding_centre_velocity_pyccel.py`` next to this file is the CPU version with the same
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
    float3 v = float3(velocities[k], velocities[n + k], velocities[2 * n + k]);
    float3 o = float3(b_old[k], b_old[n + k], b_old[2 * n + k]);
    float3 b = float3(b_new[k], b_new[n + k], b_new[2 * n + k]);
    v = mpiparticles_gc_velocity(v, o, b, v_par[k], mu[k], mass[0]);
    new_velocities[k] = v.x;
    new_velocities[n + k] = v.y;
    new_velocities[2 * n + k] = v.z;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "mpiparticles_guiding_centre_velocity",
    _SOURCE,
    inputs=["velocities", "b_old", "b_new", "v_par", "mu", "mass"],
    outputs=["new_velocities"],
    header=GUIDING_CENTRE_METAL_HEADER,
)


def guiding_centre_velocity(
    velocities: Array,
    b_old: Array,
    b_new: Array,
    v_par: Array,
    mu: Array,
    mass: float,
) -> None:
    """Replace the ``(3, N)`` velocities on the GPU, in place (see the Pyccel version)."""
    n = v_par.shape[0]
    if not n:
        return
    new_velocities = np.zeros((3, n), dtype=np.float32)
    _KERNEL(
        velocities,
        b_old,
        b_new,
        v_par,
        mu,
        np.array([mass]),
        out=(new_velocities,),
        n_threads=n,
    )
    velocities[...] = new_velocities
