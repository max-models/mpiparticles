"""The Boris velocity update shared by the kernels: NumPy (or CuPy) and Metal versions.

:func:`boris_update` is the NumPy version of ``boris.cuh`` (the CUDA device function
``mpiparticles_boris``), and :data:`BORIS_METAL_HEADER` the Metal one, the source of
``mpiparticles_boris`` to pass as ``header=`` of a Metal kernel.
:func:`relativistic_boris_update` and :func:`lorentz_factor` (with
:data:`RELATIVISTIC_BORIS_METAL_HEADER`) are the relativistic versions, on the proper velocity
``u = gamma v``. The kernels of mpiparticles use them, and so can the fused pushes of other
codes; ``boris_pyccel.py`` has the Pyccel versions, with the same arithmetic.
"""

from __future__ import annotations

import cunumpy as xp

from mpiparticles._types import Array

__all__ = [
    "BORIS_METAL_HEADER",
    "RELATIVISTIC_BORIS_METAL_HEADER",
    "boris_update",
    "lorentz_factor",
    "relativistic_boris_update",
]

#: The Metal function ``float3 mpiparticles_boris(float3 v, float3 kick, float3 t)``: the
#: velocity after the half kick, the rotation about ``t`` and the second half kick
#: (float32).
BORIS_METAL_HEADER = """
inline float3 mpiparticles_boris(float3 v, float3 kick, float3 t) {
    float3 minus = v + kick;
    float3 s = (2.0f / (1.0f + metal::dot(t, t))) * t;
    float3 prime = minus + metal::cross(minus, t);
    return minus + metal::cross(prime, s) + kick;
}
"""

#: The Metal functions ``float mpiparticles_lorentz_factor(float3 u, float inv_c2)`` and
#: ``float3 mpiparticles_relativistic_boris(float3 u, float3 kick, float3 b, float inv_c2)``:
#: the Lorentz factor of the proper velocity ``u`` and ``u`` after the half kick, the rotation
#: about ``b / gamma`` and the second half kick (float32); includes `BORIS_METAL_HEADER`.
RELATIVISTIC_BORIS_METAL_HEADER = (
    BORIS_METAL_HEADER
    + """
inline float mpiparticles_lorentz_factor(float3 u, float inv_c2) {
    return metal::sqrt(1.0f + metal::dot(u, u) * inv_c2);
}

inline float3 mpiparticles_relativistic_boris(float3 u, float3 kick, float3 b, float inv_c2) {
    float gamma = mpiparticles_lorentz_factor(u + kick, inv_c2);
    return mpiparticles_boris(u, kick, b / gamma);
}
"""
)


def _cross(a: Array, b: Array, module) -> Array:
    """Return the cross products of ``(3, N)`` columns, component by component."""
    return module.stack(
        (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
    )


def boris_update(velocities: Array, kick: Array, t: Array) -> Array:
    """Return the velocities after one Boris step; the arguments are not changed.

    Parameters
    ----------
    velocities : array
        The velocities, ``(3, N)`` (NumPy or CuPy).
    kick : array
        The half electric kick ``q/m E dt/2``, ``(3, N)`` or broadcastable to it.
    t : array
        The rotation vector ``q/m B dt/2``, ``(3, N)`` or broadcastable to it.

    Returns
    -------
    array
        The new velocities, ``(3, N)``: the half kick, the rotation about ``t`` and the
        second half kick, with the arithmetic of ``boris.cuh``.

    Examples
    --------
    >>> import numpy as np
    >>> v = boris_update(np.array([[1.0], [0.0], [0.0]]), 0.0, np.array([[0.0], [0.0], [1.0]]))
    >>> np.round(v.ravel(), 12)  # a quarter turn for |t| = 1
    array([ 0., -1.,  0.])
    """
    module = xp.get_array_module(velocities)
    t = module.broadcast_to(module.asarray(t), velocities.shape)
    minus = velocities + kick
    s = 2.0 / (1.0 + module.sum(t * t, axis=0)) * t
    prime = minus + _cross(minus, t, module)
    return minus + _cross(prime, s, module) + kick


def lorentz_factor(proper_velocities: Array, inv_c2: float) -> Array:
    """Return the Lorentz factors ``sqrt(1 + |u|^2 / c^2)`` of ``(3, N)`` proper velocities.

    Parameters
    ----------
    proper_velocities : array
        The proper velocities ``u = gamma v``, ``(3, N)`` (NumPy or CuPy).
    inv_c2 : float
        One over the speed of light squared.

    Returns
    -------
    array
        The Lorentz factors, ``(N,)``.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(lorentz_factor(np.array([[0.0], [0.0], [np.sqrt(3.0)]]), 1.0)[0]), 12)
    2.0
    """
    module = xp.get_array_module(proper_velocities)
    return module.sqrt(1.0 + module.sum(proper_velocities**2, axis=0) * inv_c2)


def relativistic_boris_update(
    proper_velocities: Array, kick: Array, b: Array, inv_c2: float
) -> Array:
    """Return the proper velocities after one relativistic Boris step; arguments unchanged.

    Parameters
    ----------
    proper_velocities : array
        The proper velocities ``u = gamma v``, ``(3, N)`` (NumPy or CuPy).
    kick : array
        The half electric kick ``q/m E dt/2``, ``(3, N)`` or broadcastable to it.
    b : array
        ``q/m B dt/2``, ``(3, N)`` or broadcastable to it; the rotation vector is ``b`` over
        the Lorentz factor of ``u + kick``.
    inv_c2 : float
        One over the speed of light squared.

    Returns
    -------
    array
        The new proper velocities, ``(3, N)``, with the arithmetic of ``boris.cuh``.

    Examples
    --------
    >>> import numpy as np
    >>> u = np.array([[1.0], [0.0], [0.0]])
    >>> new = relativistic_boris_update(u, 0.0, np.array([[0.0], [0.0], [np.sqrt(2.0)]]), 1.0)
    >>> np.round(new.ravel(), 12)  # gamma = sqrt(2): a quarter turn
    array([ 0., -1.,  0.])
    """
    module = xp.get_array_module(proper_velocities)
    b = module.broadcast_to(module.asarray(b), proper_velocities.shape)
    gamma = lorentz_factor(proper_velocities + kick, inv_c2)
    return boris_update(proper_velocities, kick, b / gamma)
