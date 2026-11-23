"""The Boris velocity update for the Pyccel kernels, inlined where they call it.

The Pyccel version of ``boris.cuh`` (CUDA) and of `mpiparticles.kernels.boris`
(NumPy and Metal), with the same arithmetic: :func:`boris_update`, and the relativistic
:func:`relativistic_boris_update` and :func:`lorentz_factor` on the proper velocity
``u = gamma v``. A Pyccel kernel, also of another package, imports them::

    from mpiparticles.kernels.boris_pyccel import boris_update

and cunumpy's kernel cache (``PyccelCache``) builds this module with the kernel; Pyccel
inlines the call. The functions have no local variables: inlined into a loop under
``#$ omp parallel for``, Pyccel would declare them outside the loop, shared by the
threads. The C compiler computes the repeated subexpressions once.
"""

from math import sqrt

from pyccel.decorators import inline


@inline
def boris_update(
    vx: float,
    vy: float,
    vz: float,
    kx: float,
    ky: float,
    kz: float,
    tx: float,
    ty: float,
    tz: float,
) -> tuple[float, float, float]:
    """Return the velocity after one Boris step: half kick, rotation, half kick.

    ``(kx, ky, kz)`` is the half electric kick ``q/m E dt/2`` and ``(tx, ty, tz)`` the
    rotation vector ``q/m B dt/2``. With ``m = v + k``, ``p = m + m x t`` and
    ``s = 2 t / (1 + |t|^2)`` the new velocity is ``m + p x s + k``.
    """
    return (
        (vx + kx)
        + (
            ((vy + ky) + (vz + kz) * tx - (vx + kx) * tz)
            * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * tz)
            - ((vz + kz) + (vx + kx) * ty - (vy + ky) * tx)
            * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * ty)
        )
        + kx,
        (vy + ky)
        + (
            ((vz + kz) + (vx + kx) * ty - (vy + ky) * tx)
            * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * tx)
            - ((vx + kx) + (vy + ky) * tz - (vz + kz) * ty)
            * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * tz)
        )
        + ky,
        (vz + kz)
        + (
            ((vx + kx) + (vy + ky) * tz - (vz + kz) * ty)
            * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * ty)
            - ((vy + ky) + (vz + kz) * tx - (vx + kx) * tz)
            * (2.0 / (1.0 + tx * tx + ty * ty + tz * tz) * tx)
        )
        + kz,
    )


@inline
def lorentz_factor(ux: float, uy: float, uz: float, inv_c2: float) -> float:
    """Return the Lorentz factor ``sqrt(1 + |u|^2 / c^2)`` of the proper velocity ``u``."""
    return sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * inv_c2)


@inline
def relativistic_boris_update(
    ux: float,
    uy: float,
    uz: float,
    kx: float,
    ky: float,
    kz: float,
    bx: float,
    by: float,
    bz: float,
    inv_c2: float,
) -> tuple[float, float, float]:
    """Return the proper velocity ``u = gamma v`` after one relativistic Boris step.

    ``(kx, ky, kz)`` is the half electric kick ``q/m E dt/2``, ``(bx, by, bz)`` is
    ``q/m B dt/2`` and ``inv_c2`` is ``1 / c^2``. The Boris step of :func:`boris_update`
    with the rotation vector ``t = b / gamma``, ``gamma`` the Lorentz factor of ``u + k``.
    """
    return boris_update(
        ux,
        uy,
        uz,
        kx,
        ky,
        kz,
        bx / sqrt(1.0 + ((ux + kx) ** 2 + (uy + ky) ** 2 + (uz + kz) ** 2) * inv_c2),
        by / sqrt(1.0 + ((ux + kx) ** 2 + (uy + ky) ** 2 + (uz + kz) ** 2) * inv_c2),
        bz / sqrt(1.0 + ((ux + kx) ** 2 + (uy + ky) ** 2 + (uz + kz) ** 2) * inv_c2),
    )
