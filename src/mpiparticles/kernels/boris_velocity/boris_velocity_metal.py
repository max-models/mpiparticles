"""The Metal version (Apple-silicon GPU, float32) of the ``boris_velocity`` kernel.

``boris_velocity_pyccel.py`` next to this file is the CPU version with the same arguments.
One thread per marker. The half step ``q/m dt / 2`` of each marker is formed on the host
in float64; the GPU rotates and kicks in float32 and the new velocities replace the old.
Import only where `cunumpy.kernels.metal_available` is true; see `mpiparticles.kernels`.
"""

from __future__ import annotations

import numpy as np

from mpiparticles._types import Array
from mpiparticles.kernels import KERNEL_PACKAGE

_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint n = half_step_shape[0];
    if (k >= n) return;
    float half = half_step[k];
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
    new_velocities[k] = vx + half * e_field[k];
    new_velocities[n + k] = vy + half * e_field[n + k];
    new_velocities[2 * n + k] = vz + half * e_field[2 * n + k];
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "mpiparticles_boris_velocity",
    _SOURCE,
    inputs=["velocities", "e_field", "b_field", "half_step"],
    outputs=["new_velocities"],
)


def boris_velocity(
    velocities: Array,
    e_field: Array,
    b_field: Array,
    charge_over_mass: Array,
    dt: Array,
) -> None:
    """Advance the ``(3, N)`` velocities by one Boris step on the GPU, in place."""
    n = velocities.shape[1]
    if not n:
        return
    half_step = 0.5 * np.asarray(charge_over_mass) * np.asarray(dt)
    new_velocities = np.zeros((3, n), dtype=np.float32)
    _KERNEL(
        velocities,
        e_field,
        b_field,
        half_step,
        out=(new_velocities,),
        n_threads=n,
    )
    velocities[...] = new_velocities
