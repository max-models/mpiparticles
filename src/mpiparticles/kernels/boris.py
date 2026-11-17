"""The Boris velocity update shared by the kernels: NumPy (or CuPy) and Metal versions.

:func:`boris_update` is the NumPy version of ``boris.cuh`` (the CUDA device function
``mpiparticles_boris``), and :data:`BORIS_METAL_HEADER` the Metal one, the source of
``mpiparticles_boris`` to pass as ``header=`` of a Metal kernel. The kernels of mpiparticles
use them, and so can the fused pushes of other codes; the Pyccel kernels have the same
arithmetic inline.
"""

from __future__ import annotations

import cunumpy as xp

from mpiparticles._types import Array

__all__ = ["BORIS_METAL_HEADER", "boris_update"]

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
