"""The Metal version (Apple-silicon GPU, float32) of the ``push_relativistic_boris`` kernel.

``push_relativistic_boris_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker. The GPU returns the new proper velocities and the
displacements ``u / gamma dt``; the host adds the displacements to the float64 positions, so
the positions are not rounded to float32. Import only where `cunumpy.kernels.metal_available`
is true; see `mpiparticles.kernels`.
"""

from __future__ import annotations

import numpy as np

from mpiparticles._types import Array
from mpiparticles.kernels import KERNEL_PACKAGE
from mpiparticles.kernels.boris import RELATIVISTIC_BORIS_METAL_HEADER

_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint n = proper_velocities_shape[1];
    if (k >= n) return;
    float step = params[0], dt = params[1], inv_c2 = params[3];
    int ndim = (int)params[2];
    float3 u = float3(proper_velocities[k], proper_velocities[n + k],
                      proper_velocities[2 * n + k]);
    float3 e = float3(e_field[k], e_field[n + k], e_field[2 * n + k]);
    float3 b = float3(b_field[k], b_field[n + k], b_field[2 * n + k]);
    u = mpiparticles_relativistic_boris(u, step * e, step * b, inv_c2);
    new_velocities[k] = u.x;
    new_velocities[n + k] = u.y;
    new_velocities[2 * n + k] = u.z;
    float move = dt / mpiparticles_lorentz_factor(u, inv_c2);
    displacement[k] = u.x * move;
    if (ndim > 1) displacement[n + k] = u.y * move;
    if (ndim > 2) displacement[2 * n + k] = u.z * move;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "mpiparticles_push_relativistic_boris",
    _SOURCE,
    inputs=["proper_velocities", "e_field", "b_field", "params"],
    outputs=["new_velocities", "displacement"],
    header=RELATIVISTIC_BORIS_METAL_HEADER,
)


def push_relativistic_boris(
    positions: Array,
    proper_velocities: Array,
    e_field: Array,
    b_field: Array,
    charge_over_mass: float,
    dt: float,
    speed_of_light: float,
) -> None:
    """Advance the markers by one relativistic Boris step on the GPU, in place."""
    ndim, n = positions.shape
    if not n:
        return
    params = np.array(
        [0.5 * charge_over_mass * dt, dt, ndim, 1.0 / speed_of_light**2],
        dtype=np.float32,
    )
    new_velocities = np.zeros((3, n), dtype=np.float32)
    displacement = np.zeros((3, n), dtype=np.float32)
    _KERNEL(
        proper_velocities,
        e_field,
        b_field,
        params,
        out=(new_velocities, displacement),
        n_threads=n,
    )
    proper_velocities[...] = new_velocities
    positions += displacement[:ndim]
