"""Pushing markers through given fields: the Boris scheme.

:func:`boris_push` advances the markers of a :class:`~mpiparticles.ParticleStore` by one
time step in electric and magnetic fields given at the markers (gathered from a grid by
the caller, e.g. with mpifield, or prescribed). It works for 1, 2 and 3 position
components with three velocity components (1D3V, 2D3V and 3D3V), in Cartesian coordinates:
the first ``ndim`` velocity components move the positions.

The step is the Boris scheme: half an electric kick, a rotation in the magnetic field and the
other half of the kick, then the drift. It is second order in time, conserves the speed in a
pure magnetic field exactly and gives the E×B drift without error; with
``omega_c dt`` large the gyration phase lags, the orbit stays on its circle. The compiled
kernel is `mpiparticles.kernels.push_boris` (Pyccel on the CPU, CUDA for CuPy arrays).
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import scope_profiler as sp

from mpiparticles._profiling import PROFILE_TAGS
from mpiparticles._types import Array
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


@sp.profile("mpiparticles.boris_push", tags=PROFILE_TAGS)
def boris_push(
    store: ParticleStore,
    e_field: Array | np.ndarray,
    b_field: Array | np.ndarray,
    dt: float,
    charge_over_mass: float,
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
    dt : float
        The time step [s].
    charge_over_mass : float
        The charge over the mass of the species [C/kg], e.g. ``-e / m_e`` for electrons.

    Raises
    ------
    ValueError
        If the store does not have three velocity components, or a field has the wrong
        shape.

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
    push_boris(positions, velocities, e, b, float(charge_over_mass), float(dt))
