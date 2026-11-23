"""The relativistic Boris update, with a charge over mass and a time step per marker.

The Pyccel version (CPU) of the ``relativistic_boris_velocity`` kernel;
``relativistic_boris_velocity_cuda.cu`` next to this file is the CUDA version (GPU), with the
same arguments. Loaded by `mpiparticles.kernels`; this file is compiled with Pyccel there, with
the block it imports (``boris_pyccel.py``, inlined).
"""

from mpiparticles.kernels.boris_pyccel import relativistic_boris_update


def relativistic_boris_velocity(
    proper_velocities: "float[:,:]",
    e_field: "float[:,:]",
    b_field: "float[:,:]",
    charge_over_mass: "float[:]",
    dt: "float[:]",
    speed_of_light: float,
) -> None:
    """Advance the proper velocities by one relativistic Boris step, in place.

    ``proper_velocities`` (``u = gamma v``), ``e_field`` and ``b_field`` are ``(3, N)``, the
    fields at the markers at the middle of the step; ``charge_over_mass`` and ``dt`` are
    ``(N,)``. The positions are not touched; see ``push_relativistic_boris``.
    """
    num_markers = proper_velocities.shape[1]
    inv_c2 = 1.0 / (speed_of_light * speed_of_light)
    # fmt: off
    #$ omp parallel for private(half, ux, uy, uz) schedule(static)
    # fmt: on
    for marker in range(num_markers):
        half = 0.5 * charge_over_mass[marker] * dt[marker]
        ux, uy, uz = relativistic_boris_update(
            proper_velocities[0, marker],
            proper_velocities[1, marker],
            proper_velocities[2, marker],
            half * e_field[0, marker],
            half * e_field[1, marker],
            half * e_field[2, marker],
            half * b_field[0, marker],
            half * b_field[1, marker],
            half * b_field[2, marker],
            inv_c2,
        )
        proper_velocities[0, marker] = ux
        proper_velocities[1, marker] = uy
        proper_velocities[2, marker] = uz
