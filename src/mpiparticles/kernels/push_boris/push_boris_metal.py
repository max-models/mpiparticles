"""The Metal version (Apple-silicon GPU, float32) of the ``push_boris`` kernel.

``push_boris_pyccel.py`` next to this file is the CPU version with the same arguments.
One thread per marker. The GPU returns the new velocities and the displacements
``v dt``; the host adds the displacements to the float64 positions, so the positions are
not rounded to float32. Import only where `cunumpy.kernels.metal_available` is true; see
`mpiparticles.kernels`.
"""

from __future__ import annotations

import numpy as np

from mpiparticles._types import Array
from mpiparticles.kernels import KERNEL_PACKAGE
from mpiparticles.kernels.boris import BORIS_METAL_HEADER

_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint n = velocities_shape[1];
    if (k >= n) return;
    float step = params[0], dt = params[1];
    int ndim = (int)params[2];
    float3 v = float3(velocities[k], velocities[n + k], velocities[2 * n + k]);
    float3 e = float3(e_field[k], e_field[n + k], e_field[2 * n + k]);
    float3 b = float3(b_field[k], b_field[n + k], b_field[2 * n + k]);
    v = mpiparticles_boris(v, step * e, step * b);
    new_velocities[k] = v.x;
    new_velocities[n + k] = v.y;
    new_velocities[2 * n + k] = v.z;
    displacement[k] = v.x * dt;
    if (ndim > 1) displacement[n + k] = v.y * dt;
    if (ndim > 2) displacement[2 * n + k] = v.z * dt;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "mpiparticles_push_boris",
    _SOURCE,
    inputs=["velocities", "e_field", "b_field", "params"],
    outputs=["new_velocities", "displacement"],
    header=BORIS_METAL_HEADER,
)


def push_boris(
    positions: Array,
    velocities: Array,
    e_field: Array,
    b_field: Array,
    charge_over_mass: float,
    dt: float,
) -> None:
    """Advance the markers by one Boris step on the GPU, in place."""
    ndim, n = positions.shape
    if not n:
        return
    params = np.array([0.5 * charge_over_mass * dt, dt, ndim], dtype=np.float32)
    new_velocities = np.zeros((3, n), dtype=np.float32)
    displacement = np.zeros((3, n), dtype=np.float32)
    _KERNEL(
        velocities,
        e_field,
        b_field,
        params,
        out=(new_velocities, displacement),
        n_threads=n,
    )
    velocities[...] = new_velocities
    positions += displacement[:ndim]
