"""Velocity moments and phase-space densities of the markers, summed over the ranks.

The moments of the markers' velocity distribution are the sums
``sum_p w_p v_p^{i_1} ... v_p^{i_k}`` of every order ``k`` up to ``max_order``: the
total weight (order 0), the momentum over the mass (order 1), ``sum w v_i v_j``
(order 2), ... `moment_indices` lists them, each symmetric moment once, and
`moment_values` gives the per-marker products, which a deposit on a grid turns
into the moments on the nodes (mpifield's ``Field.deposit_density``).
`velocity_moments` sums them over all markers of all ranks.

`phase_space_density` bins the markers on one to three phase-space axes
(`BinAxis`: ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``) and divides by the bin
volume, so the result is the phase-space density, summed over the ranks.

Examples
--------
>>> import numpy as np
>>> from mpiparticles import moment_indices, velocity_moments
>>> moment_indices(2, 2)
((), (0,), (1,), (0, 0), (0, 1), (1, 1))
>>> velocities = np.array([[1.0, -1.0], [2.0, 0.0]])
>>> velocity_moments(velocities, np.array([1.0, 3.0]), max_order=1).tolist()
[4.0, -2.0, 2.0]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any

import cunumpy as xp
import maybempi
import numpy as np
import scope_profiler as sp

from mpiparticles._profiling import PROFILE_TAGS
from mpiparticles._types import Array

__all__ = [
    "BinAxis",
    "moment_indices",
    "moment_name",
    "moment_values",
    "phase_space_density",
    "velocity_moments",
]

#: The position axes of the phase space, and the letters of the velocity components.
POSITION_AXES = ("x", "y", "z")
#: The velocity axes of the phase space.
VELOCITY_AXES = ("vx", "vy", "vz")


def moment_indices(nvel: int, max_order: int) -> tuple[tuple[int, ...], ...]:
    """Return the velocity multi-indices of the moments up to ``max_order``.

    Order by order, and within an order the indices ``(i, j, ...)`` with
    ``i <= j <= ...`` in lexicographic order: ``()``, ``(0,)``, ..., ``(0, 0)``,
    ``(0, 1)``, ... The moments are symmetric, so each is listed once:
    ``binom(nvel + k - 1, k)`` of order ``k``.
    """
    if nvel < 1:
        raise ValueError(f"nvel must be >= 1, got {nvel}")
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")
    return tuple(
        index
        for order in range(max_order + 1)
        for index in combinations_with_replacement(range(nvel), order)
    )


def moment_name(index: tuple[int, ...]) -> str:
    """Return a multi-index as the letters of its velocity components, e.g. ``"xy"``."""
    return "".join(POSITION_AXES[i] for i in index)


def _velocities_and_weights(velocities: Array, weights: Array) -> tuple[Array, Array]:
    """Return ``(nvel, N)`` velocities and ``(N,)`` weights as float arrays."""
    velocities = xp.asarray(velocities, dtype=float)
    if velocities.ndim == 1:
        velocities = velocities[None, :]
    weights = xp.asarray(weights, dtype=float).reshape(-1)
    if velocities.shape[1] != weights.shape[0]:
        raise ValueError(
            f"velocities have {velocities.shape[1]} markers, weights {weights.shape[0]}"
        )
    return velocities, weights


def moment_values(velocities: Array, weights: Array, max_order: int = 2) -> Array:
    """Return ``w v_i v_j ...`` of every marker for each of `moment_indices`.

    ``velocities`` is ``(nvel, N)`` (or ``(N,)`` for one component) and
    ``weights`` ``(N,)``; the result is ``(num_moments, N)``, on the side of the
    inputs (NumPy or CuPy). Each product reuses that of its index without the
    last component: one multiplication per moment.
    """
    velocities, weights = _velocities_and_weights(velocities, weights)
    products: dict[tuple[int, ...], Array] = {(): weights}
    indices = moment_indices(velocities.shape[0], max_order)
    for index in indices[1:]:
        products[index] = products[index[:-1]] * velocities[index[-1]]
    return xp.stack([products[index] for index in indices])


def _sum_over_ranks(array: np.ndarray, comm: Any) -> np.ndarray:
    """Sum a host array over the ranks of ``comm`` in place (no call when serial)."""
    if comm is not None and comm.Get_size() > 1:
        comm.Allreduce(maybempi.get_mpi().IN_PLACE, array, op=maybempi.get_mpi().SUM)
    return array


@sp.profile("mpiparticles.velocity_moments", tags=PROFILE_TAGS)
def velocity_moments(
    velocities: Array, weights: Array, max_order: int = 2, comm: Any = None
) -> np.ndarray:
    """Return ``sum w v_i v_j ...`` over the markers of all ranks, per `moment_indices`.

    Each rank passes its own markers (e.g. ``store.active_velocities()`` and
    ``store.active_weights()``). The result is a NumPy array of one value per
    moment, the same on every rank. Collective on ``comm``; with ``comm=None``
    or one rank no MPI call is made. Divided by the total weight (the first
    entry) the moments are the averages ``<v_i>``, ``<v_i v_j>``, ...
    """
    values = moment_values(velocities, weights, max_order)
    local = np.ascontiguousarray(xp.to_numpy(values.sum(axis=1)), dtype=float)
    return _sum_over_ranks(local, comm)


@dataclass(frozen=True)
class BinAxis:
    """One phase-space axis: ``num_bins`` equal bins on ``[lower, upper]``."""

    name: str
    lower: float
    upper: float
    num_bins: int

    def __post_init__(self) -> None:
        """Check the axis name, the range and the number of bins."""
        if self.name not in POSITION_AXES + VELOCITY_AXES:
            raise ValueError(
                f"unknown phase-space axis {self.name!r}; choose from "
                f"{POSITION_AXES + VELOCITY_AXES}"
            )
        if not self.upper > self.lower:
            raise ValueError(
                f"axis {self.name!r} needs upper > lower, "
                f"got [{self.lower}, {self.upper}]"
            )
        if int(self.num_bins) != self.num_bins or self.num_bins < 1:
            raise ValueError(f"axis {self.name!r} needs num_bins >= 1")

    @property
    def is_velocity(self) -> bool:
        """Return whether this is a velocity axis."""
        return self.name in VELOCITY_AXES

    @property
    def component(self) -> int:
        """Return the index of this axis in the position or velocity array."""
        axes = VELOCITY_AXES if self.is_velocity else POSITION_AXES
        return axes.index(self.name)

    @property
    def units(self) -> str:
        """Return the units of the axis."""
        return "m/s" if self.is_velocity else "m"

    @property
    def width(self) -> float:
        """Return the width of one bin."""
        return (self.upper - self.lower) / self.num_bins

    @property
    def edges(self) -> np.ndarray:
        """Return the ``num_bins + 1`` bin edges."""
        return np.linspace(self.lower, self.upper, int(self.num_bins) + 1)

    @property
    def centers(self) -> np.ndarray:
        """Return the ``num_bins`` bin centres."""
        edges = self.edges
        return 0.5 * (edges[:-1] + edges[1:])


def bin_axes(
    axes: Mapping[str, tuple[float, float, int]] | Sequence[BinAxis],
) -> tuple[BinAxis, ...]:
    """Return ``axes`` as `BinAxis` objects, from ``{name: (lower, upper, num_bins)}``.

    One to three axes, in the order of the dimensions of the binned array.
    """
    if isinstance(axes, Mapping):
        result = tuple(
            BinAxis(name, float(lower), float(upper), int(num_bins))
            for name, (lower, upper, num_bins) in axes.items()
        )
    else:
        result = tuple(axes)
    if not 1 <= len(result) <= 3:
        raise ValueError("a phase-space binning needs one to three axes")
    return result


@sp.profile("mpiparticles.phase_space_density", tags=PROFILE_TAGS)
def phase_space_density(
    positions: Array,
    velocities: Array,
    weights: Array,
    axes: Mapping[str, tuple[float, float, int]] | Sequence[BinAxis],
    comm: Any = None,
) -> np.ndarray:
    """Return the phase-space density of the markers of all ranks on ``axes``.

    Sums the weights in each bin and divides by the bin volume (the product of
    the bin widths), so integrating the result over the binned axes gives the
    weight inside the ranges; markers outside a range are left out.
    ``positions`` is ``(ndim, N)``, ``velocities`` ``(nvel, N)``, ``weights``
    ``(N,)``, NumPy or CuPy. The result is a NumPy array of shape
    ``(num_bins, ...)``, one dimension per axis, the same on every rank.
    Collective on ``comm``; with ``comm=None`` or one rank no MPI call is made.
    """
    axes = bin_axes(axes)
    positions = xp.asarray(positions, dtype=float)
    velocities = xp.asarray(velocities, dtype=float)
    columns = []
    for axis in axes:
        source = velocities if axis.is_velocity else positions
        if axis.component >= source.shape[0]:
            raise ValueError(
                f"axis {axis.name!r} needs {'velocity' if axis.is_velocity else 'position'}"
                f" component {axis.component}, the markers have {source.shape[0]}"
            )
        columns.append(source[axis.component])
    counts, _ = xp.histogramdd(
        xp.stack(columns, axis=-1),
        bins=[xp.asarray(axis.edges) for axis in axes],
        weights=xp.asarray(weights, dtype=float).reshape(-1),
    )
    volume = float(np.prod([axis.width for axis in axes]))
    local = np.ascontiguousarray(xp.to_numpy(counts), dtype=float) / volume
    return _sum_over_ranks(local, comm)
