"""The velocities to store after a guiding-centre step.

The Pyccel version (CPU) of the ``guiding_centre_velocity`` kernel;
``guiding_centre_velocity_cuda.cu`` next to this file is the CUDA version (GPU), with the
same arguments. Loaded by `mpiparticles.kernels`; compiled with Pyccel with the block it
imports (``guiding_centre_pyccel.py``, inlined).
"""

from mpiparticles.kernels.guiding_centre_pyccel import gc_velocity


def guiding_centre_velocity(
    velocities: "float[:,:]",
    b_old: "float[:,:]",
    b_new: "float[:,:]",
    v_par: "float[:]",
    mu: "float[:]",
    mass: float,
) -> None:
    """Replace the ``(3, N)`` velocities by ``v_par b + v_perp e_perp`` in the new field, in place.

    ``b_old`` and ``b_new`` are the magnetic field at the markers before and after the step
    ``(3, N)``; ``v_par`` the new parallel velocities and ``mu`` the magnetic moments ``(N,)``.
    """
    num_markers = v_par.shape[0]
    # fmt: off
    #$ omp parallel for private(vx, vy, vz) schedule(static)
    # fmt: on
    for marker in range(num_markers):
        vx, vy, vz = gc_velocity(
            velocities[0, marker],
            velocities[1, marker],
            velocities[2, marker],
            b_old[0, marker],
            b_old[1, marker],
            b_old[2, marker],
            b_new[0, marker],
            b_new[1, marker],
            b_new[2, marker],
            v_par[marker],
            mu[marker],
            mass,
        )
        velocities[0, marker] = vx
        velocities[1, marker] = vy
        velocities[2, marker] = vz
