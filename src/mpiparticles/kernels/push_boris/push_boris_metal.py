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

_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint n = velocities_shape[1];
    if (k >= n) return;
    float half = params[0], dt = params[1];
    int ndim = (int)params[2];
    float vx = velocities[k] + half * e_field[k];
    float vy = velocities[n + k] + half * e_field[n + k];
    float vz = velocities[2 * n + k] + half * e_field[2 * n + k];
    float tx = half * b_field[k];
    float ty = half * b_field[n + k];
    float tz = half * b_field[2 * n + k];
    float factor = 2.0f / (1.0f + tx * tx + ty * ty + tz * tz);
    float sx = factor * tx, sy = factor * ty, sz = factor * tz;
    float px = vx + vy * tz - vz * ty;
    float py = vy + vz * tx - vx * tz;
    float pz = vz + vx * ty - vy * tx;
    vx += py * sz - pz * sy;
    vy += pz * sx - px * sz;
    vz += px * sy - py * sx;
    vx += half * e_field[k];
    vy += half * e_field[n + k];
    vz += half * e_field[2 * n + k];
    new_velocities[k] = vx;
    new_velocities[n + k] = vy;
    new_velocities[2 * n + k] = vz;
    displacement[k] = vx * dt;
    if (ndim > 1) displacement[n + k] = vy * dt;
    if (ndim > 2) displacement[2 * n + k] = vz * dt;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "mpiparticles_push_boris",
    _SOURCE,
    inputs=["velocities", "e_field", "b_field", "params"],
    outputs=["new_velocities", "displacement"],
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
