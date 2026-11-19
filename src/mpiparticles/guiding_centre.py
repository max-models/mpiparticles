"""Pushing guiding centres: the drift-kinetic motion of strongly magnetized markers.

A guiding-centre marker keeps its full velocity ``v`` like any other (so loading, sources,
collisions and the kinetic energy treat it as usual): its parallel velocity is
``v_par = v . b`` and its magnetic moment ``mu = m v_perp^2 / (2 B)``, an invariant. A step
moves the guiding centre ``X`` and ``v_par`` by

    dX/dt     = v_par b + E x B / B^2 + (mu / (q B)) b x grad B + (m v_par^2 / (q B)) b x kappa,
    dv_par/dt = (q / m) E . b - (mu / m) b . grad B,

with ``kappa = (b . grad) b``, by the midpoint rule (second order; the fields are evaluated
at the start and at the midpoint), then stores ``v_par b + v_perp e_perp`` in the new field
with ``v_perp = sqrt(2 mu B / m)`` (the gyrophase carried along). Without the gyration
there is no ``omega_c dt`` limit; the polarization drift and finite-Larmor-radius effects
are left out.

:func:`guiding_centre_step` works on plain arrays, :func:`guiding_centre_push` on the
markers of a :class:`~mpiparticles.ParticleStore`. Both take the fields as a function of the
positions, ``fields(points) -> (E, B, grad_B, kappa)`` each ``(3, N)``: gathered from a grid
by the caller (mpifield) or prescribed. In 1D and 2D the axes that are not simulated are
ignorable: the drift along them is dropped. The per-marker arithmetic is the blocks of
`mpiparticles.kernels.guiding_centre` (kernels ``guiding_centre_rates`` and
``guiding_centre_velocity``: Pyccel on the CPU, CUDA for CuPy arrays), which fused pushes of
other codes can include as well.
"""

from __future__ import annotations

from collections.abc import Callable

import cunumpy as xp
import scope_profiler as sp

from mpiparticles._profiling import PROFILE_TAGS
from mpiparticles._types import Array
from mpiparticles.kernels.guiding_centre_rates import kernel as rates_kernel
from mpiparticles.kernels.guiding_centre_velocity import kernel as velocity_kernel
from mpiparticles.store import ParticleStore

__all__ = [
    "GuidingCentreFields",
    "guiding_centre_push",
    "guiding_centre_step",
    "magnetic_moment",
]

#: ``fields(points)`` with ``points`` ``(ndim, N)``: E, B, grad |B| and the curvature
#: ``(b . grad) b`` at the points, each ``(3, N)``.
GuidingCentreFields = Callable[[Array], tuple[Array, Array, Array, Array]]


def magnetic_moment(velocities: Array, b_field: Array, mass: float) -> Array:
    """Return ``mu = m v_perp^2 / (2 B)`` of the ``(3, N)`` velocities in the field ``(3, N)``.

    Examples
    --------
    >>> import numpy as np
    >>> magnetic_moment(np.array([[1.0], [2.0], [0.0]]), np.array([[2.0], [0.0], [0.0]]), 1.0)
    array([1.])
    """
    module = xp.get_array_module(velocities)
    magnitude = module.sqrt(module.sum(b_field * b_field, axis=0))
    v_par = module.sum(velocities * b_field, axis=0) / magnitude
    v_perp_squared = module.maximum(
        module.sum(velocities * velocities, axis=0) - v_par * v_par, 0.0
    )
    return 0.5 * mass * v_perp_squared / magnitude


def _c_order(values: Array, module) -> Array:
    """Return a C-ordered float copy (Pyccel checks the strides, also of a ``(3, 1)`` array)."""
    out = module.empty(values.shape, dtype=float, order="C")
    out[...] = values
    return out


def _rates(fields, points, v_par, mu, charge, mass, module):
    e, b, grad_b, curvature = (
        _c_order(module.asarray(f), module) for f in fields(points)
    )
    count = v_par.shape[0]
    drift = module.empty((3, count))
    accel = module.empty(count)
    rates_kernel(
        e, b, grad_b, curvature, v_par, mu, float(charge), float(mass), drift, accel
    )
    return drift, accel, b


@sp.profile("mpiparticles.guiding_centre_step", tags=PROFILE_TAGS)
def guiding_centre_step(
    positions: Array,
    velocities: Array,
    fields: GuidingCentreFields,
    dt: float,
    charge: float,
    mass: float,
) -> None:
    """Advance guiding centres by one midpoint step, in place.

    Parameters
    ----------
    positions : array
        The guiding centres, ``(ndim, N)`` with ``ndim`` 1, 2 or 3.
    velocities : array
        The stored velocities, ``(3, N)``: ``v_par`` and ``mu`` are taken from them, and
        they are replaced by ``v_par b + v_perp e_perp`` in the field at the new position.
    fields : callable
        ``fields(points) -> (E, B, grad_B, kappa)``, each ``(3, N)`` at the ``(ndim, N)``
        points; ``B`` without zeros.
    dt : float
        The time step [s].
    charge, mass : float
        The charge [C] and mass [kg] of the species.

    Examples
    --------
    >>> import numpy as np
    >>> def uniform(points):  # E_x = 1 V/m, B_z = 1 T: E x B drift of -1 m/s along y
    ...     n = points.shape[1]
    ...     e = np.tile([[1.0], [0.0], [0.0]], n)
    ...     b = np.tile([[0.0], [0.0], [1.0]], n)
    ...     return e, b, np.zeros((3, n)), np.zeros((3, n))
    >>> x, v = np.zeros((2, 1)), np.array([[3.0], [0.0], [0.0]])
    >>> guiding_centre_step(x, v, uniform, dt=2.0, charge=1.0, mass=1.0)
    >>> x.ravel(), np.round(np.linalg.norm(v), 12)
    (array([ 0., -2.]), np.float64(3.0))
    """
    ndim = positions.shape[0]
    if velocities.shape[0] != 3:
        raise ValueError("guiding centres need three velocity components")
    if positions.shape[1] == 0:
        return
    module = xp.get_array_module(positions)
    e, b_old, grad_b, curvature = fields(positions)
    b_old = _c_order(module.asarray(b_old), module)
    magnitude = module.sqrt(module.sum(b_old * b_old, axis=0))
    v_par = module.sum(velocities * b_old, axis=0) / magnitude
    mu = magnetic_moment(velocities, b_old, mass)
    drift, accel, _ = _rates(
        lambda _: (e, b_old, grad_b, curvature),
        positions,
        v_par,
        mu,
        charge,
        mass,
        module,
    )
    midpoint = positions + 0.5 * dt * drift[:ndim]
    drift, accel, _ = _rates(
        fields, midpoint, v_par + 0.5 * dt * accel, mu, charge, mass, module
    )
    positions += dt * drift[:ndim]
    v_par = v_par + dt * accel
    b_new = _c_order(module.asarray(fields(positions)[1]), module)
    stored = _c_order(velocities, module)
    velocity_kernel(stored, b_old, b_new, _c_order(v_par, module), mu, float(mass))
    velocities[...] = stored


@sp.profile("mpiparticles.guiding_centre_push", tags=PROFILE_TAGS)
def guiding_centre_push(
    store: ParticleStore,
    fields: GuidingCentreFields,
    dt: float,
    charge: float,
    mass: float,
) -> None:
    """Advance the alive markers of ``store`` by one guiding-centre step, in place.

    See :func:`guiding_centre_step`; the dead markers are not moved. The store needs three
    velocity components.

    Examples
    --------
    >>> import numpy as np
    >>> def mirror(points):  # B_x = 1 + x^2 along x: grad B = 2x along x
    ...     n = points.shape[1]
    ...     b = np.zeros((3, n)); b[0] = 1.0 + points[0] ** 2
    ...     g = np.zeros((3, n)); g[0] = 2.0 * points[0]
    ...     return np.zeros((3, n)), b, g, np.zeros((3, n))
    >>> store = ParticleStore(ndim=1, nvel=3)
    >>> _ = store.add(xp.asarray([[0.5]]), xp.asarray([[1.0], [1.0], [0.0]]))
    >>> guiding_centre_push(store, mirror, dt=0.01, charge=1.0, mass=1.0)
    """
    if store.nvel != 3:
        raise ValueError(
            f"guiding centres need three velocity components, the store has {store.nvel}"
        )
    length = store.length
    if length == 0:
        return
    module = xp.get_array_module(store.positions)
    rows = module.nonzero(store.alive[:length])[0]
    if rows.size == 0:
        return
    positions = _c_order(store.positions[:, rows], module)
    velocities = _c_order(store.velocities[:, rows], module)
    guiding_centre_step(positions, velocities, fields, dt, charge, mass)
    store.positions[:, rows] = positions
    store.velocities[:, rows] = velocities
