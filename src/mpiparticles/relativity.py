"""Proper velocities and Lorentz factors, for the relativistic Boris push.

The relativistic push (``relativistic=True`` of :func:`~mpiparticles.boris_push` and
:func:`~mpiparticles.boris_velocity`) works on the proper velocity ``u = gamma v``, the
momentum over the rest mass, instead of the velocity ``v``: ``u`` is unbounded while
``|v| < c``, so it keeps its precision close to the speed of light. A store pushed
relativistically holds ``u`` in ``store.velocities``; :func:`proper_velocity` turns
velocities into proper velocities to start, and :func:`coordinate_velocity` turns them back.

All arrays are component-major ``(3, N)`` (NumPy or CuPy), as in the store. The speed of
light defaults to its SI value; pass ``speed_of_light=1.0`` in normalised units.

Example
-------
>>> import numpy as np
>>> u = proper_velocity(np.array([[0.6], [0.0], [0.0]]), speed_of_light=1.0)
>>> round(float(u[0, 0]), 12), round(float(lorentz_factor(u, speed_of_light=1.0)[0]), 12)
(0.75, 1.25)
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np

from mpiparticles._types import Array

__all__ = [
    "SPEED_OF_LIGHT",
    "coordinate_velocity",
    "kinetic_energy",
    "lorentz_factor",
    "proper_velocity",
]

#: The speed of light in vacuum [m/s].
SPEED_OF_LIGHT = 299792458.0


def lorentz_factor(
    proper_velocities: Array | np.ndarray, speed_of_light: float = SPEED_OF_LIGHT
) -> Array:
    """Return the Lorentz factors ``gamma = sqrt(1 + |u|^2 / c^2)`` of proper velocities.

    Parameters
    ----------
    proper_velocities : array
        The proper velocities ``u = gamma v``, ``(3, N)`` [m/s].
    speed_of_light : float
        The speed of light, in the units of ``u``.

    Returns
    -------
    array
        The Lorentz factors, ``(N,)``.
    """
    u = xp.asarray(proper_velocities)
    module = xp.get_array_module(u)
    return module.sqrt(1.0 + module.sum(u * u, axis=0) / speed_of_light**2)


def proper_velocity(
    velocities: Array | np.ndarray, speed_of_light: float = SPEED_OF_LIGHT
) -> Array:
    """Return the proper velocities ``u = gamma v`` of velocities slower than light.

    Parameters
    ----------
    velocities : array
        The velocities ``v``, ``(3, N)``, each slower than ``speed_of_light``.
    speed_of_light : float
        The speed of light, in the units of ``v``.

    Returns
    -------
    array
        The proper velocities, ``(3, N)``.

    Raises
    ------
    ValueError
        If a velocity is not slower than light.
    """
    v = xp.asarray(velocities)
    module = xp.get_array_module(v)
    beta2 = module.sum(v * v, axis=0) / speed_of_light**2
    if bool(module.any(beta2 >= 1.0)):
        raise ValueError("every velocity must be slower than light")
    return v / module.sqrt(1.0 - beta2)


def coordinate_velocity(
    proper_velocities: Array | np.ndarray, speed_of_light: float = SPEED_OF_LIGHT
) -> Array:
    """Return the velocities ``v = u / gamma`` of proper velocities ``u``.

    Parameters
    ----------
    proper_velocities : array
        The proper velocities ``u = gamma v``, ``(3, N)``.
    speed_of_light : float
        The speed of light, in the units of ``u``.

    Returns
    -------
    array
        The velocities, ``(3, N)``, each slower than ``speed_of_light``.
    """
    u = xp.asarray(proper_velocities)
    return u / lorentz_factor(u, speed_of_light)


def kinetic_energy(
    proper_velocities: Array | np.ndarray,
    mass: float | Array | np.ndarray,
    speed_of_light: float = SPEED_OF_LIGHT,
) -> Array:
    """Return the kinetic energies ``(gamma - 1) m c^2`` of markers with proper velocities ``u``.

    Written as ``m |u|^2 / (gamma + 1)``, which keeps its precision for slow markers, where it
    tends to ``m v^2 / 2``.

    Parameters
    ----------
    proper_velocities : array
        The proper velocities ``u = gamma v``, ``(3, N)``.
    mass : float or array
        The rest mass, one for all markers or ``(N,)`` [kg].
    speed_of_light : float
        The speed of light, in the units of ``u``.

    Returns
    -------
    array
        The kinetic energies, ``(N,)`` [J].
    """
    u = xp.asarray(proper_velocities)
    module = xp.get_array_module(u)
    return mass * module.sum(u * u, axis=0) / (lorentz_factor(u, speed_of_light) + 1.0)
