"""The guiding-centre rates of every marker: ``dX/dt`` and ``dv_par/dt``.

The Pyccel version (CPU) of the ``guiding_centre_rates`` kernel; ``guiding_centre_rates_cuda.cu``
next to this file is the CUDA version (GPU), with the same arguments. Loaded by
`mpiparticles.kernels`; compiled with Pyccel with the block it imports
(``guiding_centre_pyccel.py``, inlined).
"""

from mpiparticles.kernels.guiding_centre_pyccel import gc_rates


def guiding_centre_rates(
    e_field: "float[:,:]",
    b_field: "float[:,:]",
    grad_b: "float[:,:]",
    curvature: "float[:,:]",
    v_par: "float[:]",
    mu: "float[:]",
    charge: float,
    mass: float,
    drift: "float[:,:]",
    accel: "float[:]",
) -> None:
    """Write ``dX/dt`` into ``drift`` ``(3, N)`` and ``dv_par/dt`` into ``accel`` ``(N,)``.

    The fields at the markers are ``(3, N)``: E, B (no zeros), grad |B| and the curvature
    ``(b . grad) b``; ``v_par`` and ``mu`` (the magnetic moment) are ``(N,)``.
    """
    num_markers = v_par.shape[0]
    # fmt: off
    #$ omp parallel for private(dx, dy, dz, dv) schedule(static)
    # fmt: on
    for marker in range(num_markers):
        dx, dy, dz, dv = gc_rates(
            e_field[0, marker],
            e_field[1, marker],
            e_field[2, marker],
            b_field[0, marker],
            b_field[1, marker],
            b_field[2, marker],
            grad_b[0, marker],
            grad_b[1, marker],
            grad_b[2, marker],
            curvature[0, marker],
            curvature[1, marker],
            curvature[2, marker],
            v_par[marker],
            mu[marker],
            charge,
            mass,
        )
        drift[0, marker] = dx
        drift[1, marker] = dy
        drift[2, marker] = dz
        accel[marker] = dv
