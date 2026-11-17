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
from mpiparticles.kernels.boris import BORIS_METAL_HEADER

_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint n = half_step_shape[0];
    if (k >= n) return;
    float step = half_step[k];
    float3 v = float3(velocities[k], velocities[n + k], velocities[2 * n + k]);
    float3 e = float3(e_field[k], e_field[n + k], e_field[2 * n + k]);
    float3 b = float3(b_field[k], b_field[n + k], b_field[2 * n + k]);
    v = mpiparticles_boris(v, step * e, step * b);
    new_velocities[k] = v.x;
    new_velocities[n + k] = v.y;
    new_velocities[2 * n + k] = v.z;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "mpiparticles_boris_velocity",
    _SOURCE,
    inputs=["velocities", "e_field", "b_field", "half_step"],
    outputs=["new_velocities"],
    header=BORIS_METAL_HEADER,
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
