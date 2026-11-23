"""The Metal version (Apple-silicon GPU, float32) of the ``relativistic_boris_velocity`` kernel.

``relativistic_boris_velocity_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per marker. The half step ``q/m dt / 2`` of each marker is formed on the
host in float64; the GPU kicks and rotates in float32 and the new proper velocities replace the
old. Import only where `cunumpy.kernels.metal_available` is true; see `mpiparticles.kernels`.
"""

from __future__ import annotations

import numpy as np

from mpiparticles._types import Array
from mpiparticles.kernels import KERNEL_PACKAGE
from mpiparticles.kernels.boris import RELATIVISTIC_BORIS_METAL_HEADER

_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint n = half_step_shape[0];
    if (k >= n) return;
    float step = half_step[k];
    float3 u = float3(proper_velocities[k], proper_velocities[n + k],
                      proper_velocities[2 * n + k]);
    float3 e = float3(e_field[k], e_field[n + k], e_field[2 * n + k]);
    float3 b = float3(b_field[k], b_field[n + k], b_field[2 * n + k]);
    u = mpiparticles_relativistic_boris(u, step * e, step * b, inv_c2[0]);
    new_velocities[k] = u.x;
    new_velocities[n + k] = u.y;
    new_velocities[2 * n + k] = u.z;
"""

_KERNEL = KERNEL_PACKAGE.metal_kernel(
    "mpiparticles_relativistic_boris_velocity",
    _SOURCE,
    inputs=["proper_velocities", "e_field", "b_field", "half_step", "inv_c2"],
    outputs=["new_velocities"],
    header=RELATIVISTIC_BORIS_METAL_HEADER,
)


def relativistic_boris_velocity(
    proper_velocities: Array,
    e_field: Array,
    b_field: Array,
    charge_over_mass: Array,
    dt: Array,
    speed_of_light: float,
) -> None:
    """Advance the ``(3, N)`` proper velocities by one relativistic Boris step on the GPU."""
    n = proper_velocities.shape[1]
    if not n:
        return
    half_step = 0.5 * np.asarray(charge_over_mass) * np.asarray(dt)
    inv_c2 = np.array([1.0 / speed_of_light**2], dtype=np.float32)
    new_velocities = np.zeros((3, n), dtype=np.float32)
    _KERNEL(
        proper_velocities,
        e_field,
        b_field,
        half_step,
        inv_c2,
        out=(new_velocities,),
        n_threads=n,
    )
    proper_velocities[...] = new_velocities
