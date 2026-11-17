"""One Boris step of the markers in given fields, for 1, 2 or 3 position components.

The Pyccel version (CPU) of the ``push_boris`` kernel; ``push_boris_cuda.cu`` next to
this file is the CUDA version (GPU), with the same arguments. Loaded by
`mpiparticles.kernels`; this file is compiled with Pyccel there, with the Boris update it imports
(``boris_pyccel.py``, inlined).
"""

from mpiparticles.kernels.boris_pyccel import boris_update


def push_boris(
    positions: "float[:,:]",
    velocities: "float[:,:]",
    e_field: "float[:,:]",
    b_field: "float[:,:]",
    charge_over_mass: float,
    dt: float,
) -> None:
    """Advance the markers by one Boris step, in place.

    ``positions`` is ``(ndim, N)`` with ``ndim`` 1, 2 or 3; ``velocities``, ``e_field`` and
    ``b_field`` are ``(3, N)``, the fields at the markers at the middle of the step. The
    velocity takes half the electric kick, the magnetic rotation and the other half; the
    first ``ndim`` velocity components then move the positions (Cartesian coordinates).
    """
    ndim = positions.shape[0]
    num_markers = positions.shape[1]
    half = 0.5 * charge_over_mass * dt
    # fmt: off
    #$ omp parallel for private(vx, vy, vz, d) schedule(static)
    # fmt: on
    for marker in range(num_markers):
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
        for d in range(ndim):
            positions[d, marker] = positions[d, marker] + dt * velocities[d, marker]
