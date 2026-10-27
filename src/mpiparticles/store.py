"""The per-rank container of markers (particles)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cunumpy as xp
import scope_profiler as sp

from mpiparticles._profiling import PROFILE_TAGS
from mpiparticles._types import Array
from mpiparticles.ids import IdAllocator


def _columns(array: Array, mask: Array) -> Array:
    """Return the marker columns where ``mask`` is True of a ``(ncomp, N)`` array.

    ``array[:, mask]`` comes back Fortran-ordered, which compiled kernels refuse;
    ``compress`` gives a C-contiguous copy.
    """
    return xp.get_array_module(array).compress(mask, array, axis=1)


@dataclass(frozen=True)
class StoreSnapshot:
    """A copy of everything a :class:`ParticleStore` holds, from ``snapshot()``.

    The arrays hold the first ``length`` rows (alive and dead), component-major
    like the store's own.
    """

    ndim: int
    nvel: int
    positions: Array
    velocities: Array
    weights: Array
    alive: Array
    ids: Array
    #: The id allocator's next id, so ids stay unique after a restore.
    next_id: int

    @property
    def length(self) -> int:
        """Return the number of stored rows (alive and dead)."""
        return int(self.alive.shape[0])


class ParticleStore:
    """Markers of one rank in component-major arrays of equal length.

    Storage is ``positions (ndim, capacity)``, ``velocities (nvel, capacity)``,
    ``weights (capacity,)``, ``alive (capacity,)`` (bool) and ``ids (capacity,)``
    (int64), allocated through ``cunumpy`` (NumPy or CuPy). The first ``length``
    columns are in use (the high-water mark, alive and dead markers); the rest
    is spare capacity. A dead marker (``alive`` False) stays in storage until
    :meth:`compact` removes it, so kernels that walk ``[:, :length]`` keep
    working on strided in-place views and nothing reallocates during a push.
    Compaction is the caller's decision, see :meth:`compact_if_needed`.

    Parameters
    ----------
    ndim : int
        Number of position components.
    nvel : int
        Number of velocity components, at least ``ndim`` (``nvel > ndim`` gives
        e.g. a 1D3V phase space).
    capacity : int, optional
        Initial number of columns; grows by doubling (:meth:`reserve`).
    dtype : dtype, optional
        Floating point type of positions, velocities and weights.
    id_allocator : IdAllocator, optional
        Source of marker ids. The default is the serial allocator (rank 0 of
        1); under MPI pass ``IdAllocator(rank, size)`` so that ids are unique
        across ranks.

    Examples
    --------
    >>> store = ParticleStore(ndim=1, nvel=3)
    >>> ids = store.add(xp.asarray([[0.1, 0.5]]), xp.zeros((3, 2)))
    >>> store.length, store.num_alive
    (2, 2)
    """

    def _allocate(self, capacity: int) -> None:
        self._positions = xp.zeros((self._ndim, capacity), dtype=self._dtype)
        self._velocities = xp.zeros((self._nvel, capacity), dtype=self._dtype)
        self._weights = xp.ones(capacity, dtype=self._dtype)
        self._alive = xp.zeros(capacity, dtype=bool)
        self._ids = xp.zeros(capacity, dtype=xp.int64)

    def __init__(
        self,
        ndim: int,
        nvel: int,
        *,
        capacity: int = 16,
        dtype: Any = xp.float64,
        id_allocator: IdAllocator | None = None,
    ) -> None:
        """Create an empty store."""
        if ndim < 1:
            raise ValueError("ndim must be >= 1")
        if nvel < ndim:
            raise ValueError(f"nvel ({nvel}) must be >= ndim ({ndim})")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._ndim = int(ndim)
        self._nvel = int(nvel)
        self._dtype = dtype
        self._length = 0
        self._allocator = IdAllocator() if id_allocator is None else id_allocator
        self._allocate(int(capacity))

    # -- sizes and storage --------------------------------------------------

    @property
    def ndim(self) -> int:
        """Return the number of position components."""
        return self._ndim

    @property
    def nvel(self) -> int:
        """Return the number of velocity components."""
        return self._nvel

    @property
    def dtype(self) -> Any:
        """Return the floating point type of positions, velocities and weights."""
        return self._dtype

    @property
    def length(self) -> int:
        """Return the high-water mark: columns in use, alive and dead."""
        return self._length

    @property
    def capacity(self) -> int:
        """Return the number of columns of the storage arrays."""
        return int(self._alive.shape[0])

    @property
    def num_alive(self) -> int:
        """Return the number of alive markers."""
        return int(xp.count_nonzero(self._alive[: self._length]))

    @property
    def id_allocator(self) -> IdAllocator:
        """Return the source of new marker ids."""
        return self._allocator

    @id_allocator.setter
    def id_allocator(self, allocator: IdAllocator) -> None:
        """Replace the source of new ids, for example once the rank count is known.

        The stored ids stay as they are, and the new allocator is advanced past them so that it
        never hands one out again.
        """
        allocator.advance_past(self._ids[: self._length])
        self._allocator = allocator

    @property
    def positions(self) -> Array:
        """Return the whole position storage, ``(ndim, capacity)``, not sliced.

        Slice ``[:, :length]`` for the markers in use. The array is replaced when
        storage grows, so views taken from it go stale (see :meth:`reserve`).
        """
        return self._positions

    @property
    def velocities(self) -> Array:
        """Return the whole velocity storage, ``(nvel, capacity)``, not sliced."""
        return self._velocities

    @property
    def weights(self) -> Array:
        """Return the whole weight storage, ``(capacity,)``, not sliced."""
        return self._weights

    @property
    def alive(self) -> Array:
        """Return the whole ``alive`` flags, ``(capacity,)`` bool, not sliced."""
        return self._alive

    @property
    def ids(self) -> Array:
        """Return the whole id storage, ``(capacity,)`` int64, not sliced."""
        return self._ids

    def reserve(self, n: int) -> None:
        """Make room for at least ``n`` columns, growing by doubling.

        Growth allocates new arrays and copies the first ``length`` columns, so
        ``positions``, ``velocities``, ``weights``, ``alive`` and ``ids`` are
        rebound: any view or reference taken earlier (a kernel argument, a
        ``MarkerSet`` slice) keeps pointing at the old storage and is stale. Call
        ``reserve`` before taking views, or take them again afterwards. Nothing
        happens when the capacity already suffices.
        """
        capacity = self.capacity
        if n <= capacity:
            return
        with sp.region("mpiparticles.reserve", tags=PROFILE_TAGS):
            while capacity < n:
                capacity *= 2
            old = (
                self._positions,
                self._velocities,
                self._weights,
                self._alive,
                self._ids,
            )
            self._allocate(capacity)
            end = self._length
            new = (
                self._positions,
                self._velocities,
                self._weights,
                self._alive,
                self._ids,
            )
            for target, source in zip(new, old, strict=True):
                target[..., :end] = source[..., :end]

    # -- adding and removing --------------------------------------------------

    def add(
        self,
        positions: Array,
        velocities: Array,
        weights: Array | float | None = None,
        ids: Array | None = None,
    ) -> Array:
        """Append alive markers; returns their ids.

        Parameters
        ----------
        positions : array, shape (ndim, n)
            Component-major positions.
        velocities : array, shape (nvel, n)
            Component-major velocities.
        weights : array, shape (n,) or float, optional
            Weights; default 1.
        ids : array of int, shape (n,), optional
            Ids to keep (arriving markers, a restored set). Default: new ids
            from the allocator. The allocator moves past given ids.

        Returns
        -------
        array of int64, shape (n,)
            The ids of the new markers.
        """
        positions = xp.asarray(positions, dtype=self._dtype)
        velocities = xp.asarray(velocities, dtype=self._dtype)
        if positions.ndim != 2 or positions.shape[0] != self._ndim:
            raise ValueError(
                f"positions must have shape ({self._ndim}, n), got {positions.shape}"
            )
        count = int(positions.shape[1])
        if velocities.shape != (self._nvel, count):
            raise ValueError(
                f"velocities must have shape ({self._nvel}, {count}), "
                f"got {velocities.shape}"
            )
        if weights is None:
            weights = xp.ones(count, dtype=self._dtype)
        else:
            weights = xp.asarray(weights, dtype=self._dtype)
            if weights.ndim == 0:
                weights = xp.full(count, float(weights), dtype=self._dtype)
            if weights.shape != (count,):
                raise ValueError(
                    f"weights must have shape ({count},), got {weights.shape}"
                )
        if ids is None:
            new_ids = self._allocator.allocate(count)
        else:
            new_ids = xp.asarray(ids)
            if new_ids.shape != (count,) or new_ids.dtype.kind not in "iu":
                raise ValueError(f"ids must be {count} integers, got {new_ids.shape}")
            new_ids = new_ids.astype(xp.int64)
            self._allocator.advance_past(new_ids)
        self.reserve(self._length + count)
        rows = slice(self._length, self._length + count)
        self._positions[:, rows] = positions
        self._velocities[:, rows] = velocities
        self._weights[rows] = weights
        self._alive[rows] = True
        self._ids[rows] = new_ids
        self._length += count
        return new_ids

    def deactivate(self, mask: Array) -> None:
        """Mark markers as dead; ``mask`` is a bool array over ``[:length]``.

        The rows stay in storage (with their data) until :meth:`compact`.
        """
        mask = xp.asarray(mask, dtype=bool)
        if mask.shape != (self._length,):
            raise ValueError(
                f"mask must have shape ({self._length},), got {mask.shape}"
            )
        self._alive[: self._length] &= ~mask

    def _clear_rows(self, start: int, stop: int) -> None:
        """Reset columns ``start:stop`` to the empty state."""
        self._positions[:, start:stop] = 0.0
        self._velocities[:, start:stop] = 0.0
        self._weights[start:stop] = 1.0
        self._alive[start:stop] = False
        self._ids[start:stop] = 0

    def remove(self, index: int) -> None:
        """Remove the marker at ``index`` (alive or dead), shifting the rest down."""
        if not 0 <= index < self._length:
            raise IndexError("marker index out of range")
        stop = self._length
        for array in (self._positions, self._velocities):
            array[:, index : stop - 1] = array[:, index + 1 : stop].copy()
        for array in (self._weights, self._alive, self._ids):
            array[index : stop - 1] = array[index + 1 : stop].copy()
        self._clear_rows(stop - 1, stop)
        self._length -= 1

    def clear(self) -> None:
        """Remove all markers; the id counter keeps running (ids are never reused)."""
        self._clear_rows(0, self._length)
        self._length = 0

    def _keep_first(self, count: int) -> None:
        """Keep the first ``count`` columns, all alive, and clear the rest."""
        self._alive[:count] = True
        self._clear_rows(count, self._length)
        self._length = count

    @sp.profile("mpiparticles.compact", tags=PROFILE_TAGS)
    def compact(self) -> None:
        """Move the alive markers to the front, in order, and drop the dead ones."""
        end = self._length
        count = xp.algorithms.compact_by_mask(
            self._alive[:end],
            self._positions[:, :end],
            self._velocities[:, :end],
            self._weights[:end],
            self._ids[:end],
            axis=-1,
        )
        self._keep_first(count)

    def compact_if_needed(self, max_dead_fraction: float = 0.25) -> bool:
        """Compact once the dead markers exceed a fraction of ``length``.

        Returns whether it compacted. Compacting on a fraction keeps the cost of
        walking dead rows amortized.
        """
        if self._length == 0:
            return False
        dead = self._length - self.num_alive
        if dead <= max_dead_fraction * self._length:
            return False
        self.compact()
        return True

    def sort_by_key(self, keys: Array) -> None:
        """Reorder all fields by ``keys`` (stable) and drop the dead markers.

        Parameters
        ----------
        keys : array of int, shape (length,)
            One key per stored column, for example the cell index. Keys of dead
            markers are ignored. Ids move as int64, never as floats.
        """
        end = self._length
        keys = xp.asarray(keys)
        if keys.shape != (end,):
            raise ValueError(f"keys must have shape ({end},), got {keys.shape}")
        with sp.region("mpiparticles.sort_by_key", tags=PROFILE_TAGS):
            alive = self._alive[:end]
            count = int(xp.count_nonzero(alive))
            if 0 < count < end:
                top = keys[alive].max()
                if keys.dtype.kind in "iu" and top < xp.iinfo(keys.dtype).max:
                    # the dead sort last, behind a key above every alive one
                    keys = xp.where(alive, keys, top + 1)
                else:
                    keys = keys[alive]
                    self.compact()
                    end = count
            if count:
                fields = (
                    self._positions[:, :end],
                    self._velocities[:, :end],
                    self._weights[:end],
                    self._ids[:end],
                )
                # component-major: cunumpy gathers the (ncomp, n) arrays row by row
                _, _, *sorted_fields = xp.algorithms.sort_by_key(keys, *fields, axis=-1)
                for field, values in zip(fields, sorted_fields, strict=True):
                    field[..., :count] = values[..., :count]
            self._keep_first(count)

    # -- alive markers --------------------------------------------------------

    def active_positions(self) -> Array:
        """Return a copy of the positions of the alive markers, ``(ndim, n)``."""
        return _columns(self._positions[:, : self._length], self._alive[: self._length])

    def active_velocities(self) -> Array:
        """Return a copy of the velocities of the alive markers, ``(nvel, n)``."""
        return _columns(
            self._velocities[:, : self._length], self._alive[: self._length]
        )

    def active_weights(self) -> Array:
        """Return a copy of the weights of the alive markers, ``(n,)``."""
        return self._weights[: self._length][self._alive[: self._length]]

    def active_ids(self) -> Array:
        """Return a copy of the ids of the alive markers, ``(n,)``."""
        return self._ids[: self._length][self._alive[: self._length]]

    def set_active_positions(self, positions: Array) -> None:
        """Set the positions of the alive markers from ``(ndim, num_alive)``."""
        self._positions[:, : self._length][:, self._alive[: self._length]] = positions

    def set_active_velocities(self, velocities: Array) -> None:
        """Set the velocities of the alive markers from ``(nvel, num_alive)``."""
        self._velocities[:, : self._length][:, self._alive[: self._length]] = velocities

    def set_active_weights(self, weights: Array) -> None:
        """Set the weights of the alive markers from ``(num_alive,)``."""
        self._weights[: self._length][self._alive[: self._length]] = weights

    # -- checkpointing --------------------------------------------------------

    @sp.profile("mpiparticles.snapshot", tags=PROFILE_TAGS)
    def snapshot(self) -> StoreSnapshot:
        """Return a copy of all markers (alive and dead) and the id counter."""
        end = self._length
        return StoreSnapshot(
            ndim=self._ndim,
            nvel=self._nvel,
            positions=self._positions[:, :end].copy(),
            velocities=self._velocities[:, :end].copy(),
            weights=self._weights[:end].copy(),
            alive=self._alive[:end].copy(),
            ids=self._ids[:end].copy(),
            next_id=self._allocator.next_id,
        )

    def restore(self, snapshot: StoreSnapshot) -> None:
        """Replace the contents by a snapshot, including the id counter.

        Storage grows if needed (see :meth:`reserve`) and is never shrunk.
        """
        if (snapshot.ndim, snapshot.nvel) != (self._ndim, self._nvel):
            raise ValueError(
                f"snapshot is for ndim={snapshot.ndim}, nvel={snapshot.nvel}, "
                f"this store has ndim={self._ndim}, nvel={self._nvel}"
            )
        with sp.region("mpiparticles.restore", tags=PROFILE_TAGS):
            end = snapshot.length
            self.reserve(end)
            self._clear_rows(0, self._length)
            self._positions[:, :end] = snapshot.positions
            self._velocities[:, :end] = snapshot.velocities
            self._weights[:end] = snapshot.weights
            self._alive[:end] = snapshot.alive
            self._ids[:end] = snapshot.ids
            self._length = end
            self._allocator.next_id = snapshot.next_id

    # -- plasmacoll's MarkerSet -----------------------------------------------

    def as_marker_set(self) -> MarkerView:
        """Return a ``plasmacoll.markers.MarkerSet`` view of this store.

        The view is marker-major, ``positions (N, ndim)``, ``velocities (N,
        nvel)``, ``weights (N,)``: transposed views of the storage, not copies,
        so in-place updates (collisions change ``velocities``) land in the
        store. It needs a store without dead rows, so this call compacts the
        store first, and the view compacts again when markers were deactivated
        since. ``remove(mask)`` and ``add(...)`` work on the store.
        """
        view = MarkerView(self)
        view._ready()  # compact now
        return view


class MarkerView:
    """Marker-major, non-copy view of a :class:`ParticleStore`.

    Implements plasmacoll's ``MarkerSet`` protocol (``positions``,
    ``velocities``, ``weights``, ``remove``, ``add``) without importing
    plasmacoll. Make it with :meth:`ParticleStore.as_marker_set`. The arrays
    are views of the store's current storage; after ``add`` or ``remove`` (which
    may reallocate or reorder) read the properties again.
    """

    def __init__(self, store: ParticleStore) -> None:
        """Wrap ``store``."""
        self._store = store

    def _ready(self) -> ParticleStore:
        store = self._store
        if store.num_alive != store.length:
            store.compact()
        return store

    @property
    def positions(self) -> Array:
        """Return the positions, shape ``(N, ndim)``, a view."""
        store = self._ready()
        return store.positions[:, : store.length].T

    @property
    def velocities(self) -> Array:
        """Return the velocities, shape ``(N, nvel)``, a view updated in place."""
        store = self._ready()
        return store.velocities[:, : store.length].T

    @property
    def weights(self) -> Array:
        """Return the weights, shape ``(N,)``, a view."""
        store = self._ready()
        return store.weights[: store.length]

    def remove(self, mask: Array) -> None:
        """Remove the markers flagged in the bool ``mask`` of length ``N``."""
        store = self._ready()
        store.deactivate(mask)
        store.compact()

    def add(self, positions: Array, velocities: Array, weights: Array) -> None:
        """Append markers given marker-major: ``(n, ndim)``, ``(n, nvel)``, ``(n,)``."""
        positions = xp.asarray(positions)
        velocities = xp.asarray(velocities)
        self._ready().add(positions.T, velocities.T, weights)
