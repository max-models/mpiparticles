r"""The guiding-centre blocks shared by the kernels: NumPy (or CuPy) and Metal versions.

A guiding-centre (drift-kinetic) marker is its guiding centre ``X``, its parallel velocity
``v_par`` and its magnetic moment ``mu = m v_perp^2 / (2 B)``. Two blocks give its motion:

- :func:`guiding_centre_rates`: ``dX/dt`` and ``dv_par/dt`` at the marker,

  .. math::

     \dot{\mathbf X} = v_\parallel \mathbf b + \frac{\mathbf E\times\mathbf B}{B^2}
       + \frac{\mu}{qB}\,\mathbf b\times\nabla B
       + \frac{m v_\parallel^2}{qB}\,\mathbf b\times\boldsymbol\kappa, \qquad
     \dot v_\parallel = \frac{q}{m}\mathbf E\cdot\mathbf b
       - \frac{\mu}{m}\,\mathbf b\cdot\nabla B,

  with ``b = B / B`` and the curvature ``kappa = (b . grad) b``;
- :func:`guiding_centre_velocity`: the full velocity a code stores for the marker after a
  step, ``v_par b + v_perp e_perp`` in the new field, with ``v_perp = sqrt(2 mu B / m)``
  and ``e_perp`` the old perpendicular velocity projected on the plane normal to the new
  ``b`` (the gyrophase carried along).

``guiding_centre.cuh`` is the CUDA version (device functions ``mpiparticles_gc_rates`` and
``mpiparticles_gc_velocity``), :data:`GUIDING_CENTRE_METAL_HEADER` the Metal one and
``guiding_centre_pyccel.py`` the Pyccel one, all with the same arithmetic. The kernels
``guiding_centre_rates`` and ``guiding_centre_velocity`` and the schemes of
`mpiparticles.guiding_centre` use them, and so can the fused pushes of other codes.
"""

from __future__ import annotations

import cunumpy as xp

from mpiparticles._types import Array

__all__ = [
    "GUIDING_CENTRE_METAL_HEADER",
    "guiding_centre_rates",
    "guiding_centre_velocity",
]

#: The Metal functions ``mpiparticles_gc_rates`` (returns ``float4``: ``dX/dt`` and
#: ``dv_par/dt``) and ``mpiparticles_gc_velocity`` (returns the new ``float3`` velocity),
#: float32.
GUIDING_CENTRE_METAL_HEADER = """
inline float4 mpiparticles_gc_rates(float3 e, float3 b, float3 grad_b, float3 kappa,
                                    float v_par, float mu, float charge, float mass) {
    float magnitude = metal::length(b);
    float3 unit = b / magnitude;
    float3 drift = v_par * unit
        + metal::cross(e, b) / (magnitude * magnitude)
        + (mu / (charge * magnitude)) * metal::cross(unit, grad_b)
        + (mass * v_par * v_par / (charge * magnitude)) * metal::cross(unit, kappa);
    float accel = (charge / mass) * metal::dot(e, unit) - (mu / mass) * metal::dot(unit, grad_b);
    return float4(drift, accel);
}

inline float3 mpiparticles_gc_velocity(float3 v, float3 b_old, float3 b_new,
                                       float v_par, float mu, float mass) {
    float3 old_unit = b_old / metal::length(b_old);
    float magnitude = metal::length(b_new);
    float3 unit = b_new / magnitude;
    float3 perp = v - metal::dot(v, old_unit) * old_unit;
    float3 direction = perp - metal::dot(perp, unit) * unit;
    float norm = metal::max(metal::length(direction), 1.0e-30f);
    float v_perp = metal::sqrt(metal::max(2.0f * mu * magnitude / mass, 0.0f));
    return v_par * unit + (v_perp / norm) * direction;
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


def guiding_centre_rates(
    e_field: Array,
    b_field: Array,
    grad_b: Array,
    curvature: Array,
    v_par: Array,
    mu: Array,
    charge: float,
    mass: float,
) -> tuple[Array, Array]:
    """Return ``dX/dt`` ``(3, N)`` and ``dv_par/dt`` ``(N,)`` of guiding centres.

    Parameters
    ----------
    e_field, b_field : array
        The electric and magnetic fields at the markers, ``(3, N)``; ``B`` has no zeros.
    grad_b : array
        The gradient of ``|B|`` at the markers, ``(3, N)``.
    curvature : array
        The curvature ``(b . grad) b`` at the markers, ``(3, N)``.
    v_par : array
        The parallel velocities, ``(N,)``.
    mu : array
        The magnetic moments ``m v_perp^2 / (2 B)``, ``(N,)``.
    charge, mass : float
        The charge and mass of the species.

    Returns
    -------
    drift : array
        ``dX/dt``, ``(3, N)``: parallel motion, E x B, grad-B and curvature drifts.
    accel : array
        ``dv_par/dt``, ``(N,)``: the parallel electric force and the mirror force.

    Examples
    --------
    >>> import numpy as np
    >>> e = np.array([[1.0], [0.0], [0.0]])
    >>> b = np.array([[0.0], [0.0], [2.0]])
    >>> zero = np.zeros((3, 1))
    >>> drift, accel = guiding_centre_rates(e, b, zero, zero, np.zeros(1), np.zeros(1), 1.0, 1.0)
    >>> drift.ravel()  # E x B / B^2
    array([ 0. , -0.5,  0. ])
    """
    module = xp.get_array_module(b_field)
    magnitude = module.sqrt(module.sum(b_field * b_field, axis=0))
    unit = b_field / magnitude
    drift = (
        v_par * unit
        + _cross(e_field, b_field, module) / (magnitude * magnitude)
        + (mu / (charge * magnitude)) * _cross(unit, grad_b, module)
        + (mass * v_par * v_par / (charge * magnitude))
        * _cross(unit, curvature, module)
    )
    accel = (charge / mass) * module.sum(e_field * unit, axis=0) - (
        mu / mass
    ) * module.sum(unit * grad_b, axis=0)
    return drift, accel


def guiding_centre_velocity(
    velocities: Array,
    b_old: Array,
    b_new: Array,
    v_par: Array,
    mu: Array,
    mass: float,
) -> Array:
    """Return the velocities ``(3, N)`` to store after a guiding-centre step.

    Parameters
    ----------
    velocities : array
        The stored velocities before the step, ``(3, N)``.
    b_old, b_new : array
        The magnetic field at the markers before and after the step, ``(3, N)``.
    v_par : array
        The new parallel velocities, ``(N,)``.
    mu : array
        The magnetic moments, ``(N,)``.
    mass : float
        The mass of the species.

    Returns
    -------
    array
        ``v_par b + v_perp e_perp`` with ``v_perp = sqrt(2 mu B / m)`` in the new field and
        ``e_perp`` the old perpendicular velocity projected on the plane normal to the new
        ``b``. A marker without gyration (``mu = 0``) gets no perpendicular velocity; one
        whose old perpendicular velocity lies along the new ``b`` (a field turned by 90
        degrees in one step) loses it.
    """
    module = xp.get_array_module(velocities)
    old_unit = b_old / module.sqrt(module.sum(b_old * b_old, axis=0))
    magnitude = module.sqrt(module.sum(b_new * b_new, axis=0))
    unit = b_new / magnitude
    perp = velocities - module.sum(velocities * old_unit, axis=0) * old_unit
    direction = perp - module.sum(perp * unit, axis=0) * unit
    norm = module.maximum(
        module.sqrt(module.sum(direction * direction, axis=0)), 1.0e-300
    )
    v_perp = module.sqrt(module.maximum(2.0 * mu * magnitude / mass, 0.0))
    return v_par * unit + (v_perp / norm) * direction
