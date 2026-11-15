"""Pushing markers through given fields: the Boris scheme.

:func:`boris_push` advances the markers of a :class:`~mpiparticles.ParticleStore` by one
time step in electric and magnetic fields given at the markers (gathered from a grid by
the caller, e.g. with mpifield, or prescribed). It works for 1, 2 and 3 position
components with three velocity components (1D3V, 2D3V and 3D3V), in Cartesian coordinates:
the first ``ndim`` velocity components move the positions. The time step and the charge
over mass are one number for all markers or one per marker (several species in one store,
a charge state per marker, or a step per marker such as a fixed path length).

:func:`boris_velocity` is the velocity update alone, on plain ``(3, N)`` arrays: for codes
that keep their markers in their own arrays and move the positions themselves (a step
tested against walls, the half step back that starts the leapfrog).

The step is the Boris scheme: half an electric kick, a rotation in the magnetic field and the
other half of the kick, then the drift. It is second order in time, conserves the speed in a
pure magnetic field exactly and gives the E×B drift without error; with
``omega_c dt`` large the gyration phase lags, the orbit stays on its circle. The compiled
kernels are `mpiparticles.kernels.push_boris` and `mpiparticles.kernels.boris_velocity`
(Pyccel on the CPU, CUDA for CuPy arrays).
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import scope_profiler as sp

from mpiparticles._profiling import PROFILE_TAGS
from mpiparticles._types import Array
from mpiparticles.kernels.boris_velocity import kernel as boris_velocity_kernel
from mpiparticles.kernels.push_boris import kernel as push_boris
from mpiparticles.store import ParticleStore


def _at_markers(
    field: Array | np.ndarray, length: int, name: str, like: Array
) -> Array:
    """Return a field at the markers as a C-contiguous ``(3, length)`` array.

    A ``(3,)`` field is the same at every marker.
    """
    module = xp.get_array_module(like)
    values = module.asarray(field, dtype=like.dtype)
    if values.shape == (3,):
        values = module.broadcast_to(values[:, None], (3, length))
    if values.shape != (3, length):
        raise ValueError(
            f"{name} must be (3,) or (3, {length}) (one column per stored marker), "
            f"not {values.shape}"
        )
    return module.ascontiguousarray(values)


def _per_marker(
    value: float | Array | np.ndarray, length: int, name: str, like: Array
) -> float | Array:
    """Return a scalar as a float, or a per-marker value as a C-contiguous ``(length,)`` array."""
    if np.ndim(value) == 0:
        return float(value)
    module = xp.get_array_module(like)
    values = module.asarray(value, dtype=like.dtype)
    if values.shape != (length,):
        raise ValueError(
            f"{name} must be a number or ({length},) (one per stored marker), "
            f"not {values.shape}"
        )
    return module.ascontiguousarray(values)


def _full(value: float | Array, length: int, like: Array) -> Array:
    """Return ``value`` as a ``(length,)`` array, a scalar repeated."""
    if isinstance(value, float):
        return xp.get_array_module(like).full(length, value, dtype=like.dtype)
    return value


@sp.profile("mpiparticles.boris_velocity", tags=PROFILE_TAGS)
def boris_velocity(
    velocities: Array | np.ndarray,
    e_field: Array | np.ndarray,
    b_field: Array | np.ndarray,
    dt: float | Array | np.ndarray,
    charge_over_mass: float | Array | np.ndarray,
) -> None:
    """Advance velocities by one Boris step in place, without moving any positions.

    The kick-rotate-kick of :func:`boris_push` on a plain array. With a negative ``dt`` it
    steps back, e.g. half a step to start the leapfrog from the initial positions.

    Parameters
    ----------
    velocities : array
        The velocities, a float64 ``(3, N)`` array (NumPy or CuPy); changed in place.
    e_field : array
        The electric field at the markers at the middle of the step, ``(3, N)``, or
        ``(3,)`` for the same field at every marker [V/m].
    b_field : array
        The magnetic field at the markers, ``(3, N)`` or ``(3,)`` [T].
    dt : float or array
        The time step [s], one for all markers or ``(N,)``.
    charge_over_mass : float or array
        The charge over the mass [C/kg], one for all markers or ``(N,)``.

    Raises
    ------
    ValueError
        If ``velocities`` is not a float64 ``(3, N)`` array, or a field, ``dt``
        or ``charge_over_mass`` has the wrong shape.

    Examples
    --------
    >>> v = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    >>> boris_velocity(v, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.1, np.array([1.0, -1.0]))
    >>> bool(v[1, 0] < 0.0 < v[1, 1])  # the two charges turn opposite ways
    True
    """
    shape = getattr(velocities, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != 3:
        raise ValueError(f"velocities must be a (3, N) array, not {shape}")
    if velocities.dtype != np.float64:
        raise ValueError(f"velocities must be float64, not {velocities.dtype}")
    length = shape[1]
    if length == 0:
        return
    e = _at_markers(e_field, length, "e_field", velocities)
    b = _at_markers(b_field, length, "b_field", velocities)
    step = _full(_per_marker(dt, length, "dt", velocities), length, velocities)
    qm = _full(
        _per_marker(charge_over_mass, length, "charge_over_mass", velocities),
        length,
        velocities,
    )
    boris_velocity_kernel(velocities, e, b, qm, step)


@sp.profile("mpiparticles.boris_push", tags=PROFILE_TAGS)
def boris_push(
    store: ParticleStore,
    e_field: Array | np.ndarray,
    b_field: Array | np.ndarray,
    dt: float | Array | np.ndarray,
    charge_over_mass: float | Array | np.ndarray,
) -> None:
    """Advance every stored marker by one Boris step, in place.

    Parameters
    ----------
    store : ParticleStore
        The markers: ``ndim`` 1, 2 or 3 and ``nvel`` 3. All ``store.length`` stored
        columns are pushed; a dead marker's column moves too and stays dead.
    e_field : array
        The electric field at the markers at the middle of the step, ``(3, length)``, or
        ``(3,)`` for the same field at every marker [V/m].
    b_field : array
        The magnetic field at the markers, ``(3, length)`` or ``(3,)`` [T].
    dt : float or array
        The time step [s], one for all markers or ``(length,)``, one per stored column.
    charge_over_mass : float or array
        The charge over the mass [C/kg], e.g. ``-e / m_e`` for electrons; one for all
        markers or ``(length,)``.

    Raises
    ------
    ValueError
        If the store does not have three velocity components, or a field, ``dt`` or
        ``charge_over_mass`` has the wrong shape.

    Examples
    --------
    >>> store = ParticleStore(ndim=2, nvel=3)
    >>> _ = store.add(xp.asarray([[0.0], [0.0]]), xp.asarray([[1.0], [0.0], [0.0]]))
    >>> boris_push(store, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], dt=0.1, charge_over_mass=1.0)
    """
    if store.nvel != 3:
        raise ValueError(
            f"the Boris push needs three velocity components, the store has {store.nvel}"
        )
    length = store.length
    if length == 0:
        return
    positions = store.positions[:, :length]
    velocities = store.velocities[:, :length]
    e = _at_markers(e_field, length, "e_field", positions)
    b = _at_markers(b_field, length, "b_field", positions)
    step = _per_marker(dt, length, "dt", positions)
    qm = _per_marker(charge_over_mass, length, "charge_over_mass", positions)
    if isinstance(step, float) and isinstance(qm, float):
        push_boris(positions, velocities, e, b, qm, step)
        return
    # per-marker values: the velocity kernel, then the drift
    boris_velocity_kernel(
        velocities,
        e,
        b,
        _full(qm, length, positions),
        _full(step, length, positions),
    )
    positions += step * velocities[: store.ndim]
