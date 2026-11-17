"""The Boris velocity update, with a charge over mass and a time step per marker.

The Pyccel version (CPU) of the ``boris_velocity`` kernel; ``boris_velocity_cuda.cu`` next
to this file is the CUDA version (GPU), with the same arguments. Loaded by
`mpiparticles.kernels`; this file is compiled with Pyccel there, with the Boris update it imports
(``boris_pyccel.py``, inlined).
"""

from mpiparticles.kernels.boris_pyccel import boris_update


def boris_velocity(
    velocities: "float[:,:]",
    e_field: "float[:,:]",
    b_field: "float[:,:]",
    charge_over_mass: "float[:]",
    dt: "float[:]",
) -> None:
    """Advance the velocities by one Boris step, in place; the positions are not touched.

    ``velocities``, ``e_field`` and ``b_field`` are ``(3, N)``, the fields at the markers at
    the middle of the step; ``charge_over_mass`` and ``dt`` are ``(N,)``. The velocity takes
    half the electric kick, the magnetic rotation and the other half, as in ``push_boris``.
    """
    num_markers = velocities.shape[1]
    # fmt: off
    #$ omp parallel for private(half, vx, vy, vz) schedule(static)
    # fmt: on
    for marker in range(num_markers):
        half = 0.5 * charge_over_mass[marker] * dt[marker]
        vx, vy, vz = boris_update(
            velocities[0, marker],
            velocities[1, marker],
            velocities[2, marker],
            half * e_field[0, marker],
            half * e_field[1, marker],
            half * e_field[2, marker],
            half * b_field[0, marker],
            half * b_field[1, marker],
            half * b_field[2, marker],
        )
        velocities[0, marker] = vx
        velocities[1, marker] = vy
        velocities[2, marker] = vz
