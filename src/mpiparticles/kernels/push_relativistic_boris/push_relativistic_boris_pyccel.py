"""One relativistic Boris step of the markers in given fields, for 1, 2 or 3 position components.

The Pyccel version (CPU) of the ``push_relativistic_boris`` kernel;
``push_relativistic_boris_cuda.cu`` next to this file is the CUDA version (GPU), with the same
arguments. Loaded by `mpiparticles.kernels`; this file is compiled with Pyccel there, with the
blocks it imports (``boris_pyccel.py``, inlined).
"""

from mpiparticles.kernels.boris_pyccel import lorentz_factor, relativistic_boris_update


def push_relativistic_boris(
    positions: "float[:,:]",
    proper_velocities: "float[:,:]",
    e_field: "float[:,:]",
    b_field: "float[:,:]",
    charge_over_mass: float,
    dt: float,
    speed_of_light: float,
) -> None:
    """Advance the markers by one relativistic Boris step, in place.

    ``positions`` is ``(ndim, N)`` with ``ndim`` 1, 2 or 3; ``proper_velocities`` (``u =
    gamma v``), ``e_field`` and ``b_field`` are ``(3, N)``, the fields at the markers at the
    middle of the step. The proper velocity takes half the electric kick, the magnetic rotation
    (by ``q/m B dt / gamma``) and the other half; the first ``ndim`` components of the new
    velocity ``u / gamma`` then move the positions (Cartesian coordinates).
    """
    ndim = positions.shape[0]
    num_markers = positions.shape[1]
    half = 0.5 * charge_over_mass * dt
    inv_c2 = 1.0 / (speed_of_light * speed_of_light)
    # fmt: off
    #$ omp parallel for private(ux, uy, uz, step, d) schedule(static)
    # fmt: on
    for marker in range(num_markers):
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
        step = dt / lorentz_factor(ux, uy, uz, inv_c2)
        for d in range(ndim):
            positions[d, marker] = (
                positions[d, marker] + step * proper_velocities[d, marker]
            )
