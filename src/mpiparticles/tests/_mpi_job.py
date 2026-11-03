"""The MPI job of ``test_mpi.py``: run with ``mpiexec -n N python _mpi_job.py [big]``.

Every rank checks its part and the job aborts on any failure.
"""

from __future__ import annotations

import sys
import traceback

import cunumpy as xp
import mpiarray
import numpy as np
from mpi4py import MPI

import mpiparticles
from mpiparticles import (
    BoxDecomposition,
    IdAllocator,
    ParticleStore,
    gather_columns,
    global_count,
    migrate,
    phase_space_density,
    velocity_moments,
)

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
PROC_SIZES = {2: (2, 1), 4: (2, 2)}[SIZE]
NMARKERS = 200


def host(array):
    return np.asarray(xp.to_numpy(array))


def weight_of(ids):
    return (ids % 9973).astype(float) * 0.5 + 1.0


def velocity_of(ids):
    base = (ids % 9973).astype(float)
    return np.stack([base, -base, base * 0.25])


def make_store(deco, big: bool):
    first = ((2**60 // SIZE) * SIZE + RANK) if big else RANK
    store = ParticleStore(
        2, 3, capacity=8, id_allocator=IdAllocator(RANK, SIZE, next_id=first)
    )
    if RANK == SIZE - 1:
        return store  # one rank starts without markers
    rng = np.random.default_rng(RANK)
    low, high = deco.bounds()
    own = rng.uniform(low, high, size=(NMARKERS - 50, 2)).T
    anywhere = rng.uniform(0.0, 1.0, size=(50, 2)).T
    positions = np.concatenate([own, anywhere], axis=1)
    ids = store.add(xp.asarray(positions), xp.zeros((3, NMARKERS)))
    store.velocities[:, :NMARKERS] = xp.asarray(velocity_of(host(ids)))
    store.weights[:NMARKERS] = xp.asarray(weight_of(host(ids)))
    return store


def check_state(store, deco, expected_ids):
    """Conservation, ownership, unique ids, and data that follows the ids."""
    assert global_count(store, COMM) == len(expected_ids)
    alive = store.alive[: store.length]
    positions = store.positions[:, : store.length]
    assert bool(xp.all(~alive | deco.owned_mask(positions)))
    assert bool(xp.all(~alive | (deco.owners(positions) == RANK)))
    ids = gather_columns(store.active_ids(), COMM, root=None)
    assert ids.dtype == np.int64
    assert len(np.unique(ids)) == len(ids), "ids are not unique"
    assert set(ids.tolist()) == expected_ids, "ids were lost or invented"
    mine = host(store.active_ids())
    assert np.array_equal(host(store.active_weights()), weight_of(mine))
    assert np.array_equal(host(store.active_velocities()), velocity_of(mine))


def main() -> None:
    big = "big" in sys.argv[1:]
    deco = BoxDecomposition(COMM, [0.0, 0.0], [1.0, 1.0], PROC_SIZES)
    assert isinstance(deco, mpiparticles.Decomposition)
    store = make_store(deco, big)
    ids_before = set(gather_columns(store.active_ids(), COMM, root=None).tolist())
    assert len(ids_before) == (SIZE - 1) * NMARKERS
    if big:
        assert min(ids_before) > 2**53

    # record what goes through the exchange
    traffic = []
    original = mpiarray.migrate

    def recording(destinations, *arrays, **kwargs):
        traffic.append(
            (
                len(destinations),
                [(a.shape, a.dtype) for a in arrays],
                host(destinations),
            )
        )
        return original(destinations, *arrays, **kwargs)

    mpiarray.migrate = recording
    try:
        expected_leavers = int(
            xp.count_nonzero(
                store.alive[: store.length]
                & (deco.owners(store.positions[:, : store.length]) != RANK)
            )
        )
        length_before = store.length
        sent, received = migrate(store, deco)
    finally:
        mpiarray.migrate = original

    # only the leavers were sent, as float rows plus an int64 id array
    assert sent == expected_leavers, (sent, expected_leavers)
    assert len(traffic) == 1
    rows, arrays, destinations = traffic[0]
    assert rows == sent
    assert sent < length_before or length_before == 0
    assert arrays[0][0] == (sent, 2 + 3 + 1) and arrays[0][1] == np.float64
    assert arrays[1][0] == (sent,) and arrays[1][1] == np.int64
    assert len(destinations) == sent and destinations.max(initial=0) < SIZE
    total_sent = COMM.allreduce(sent)
    assert total_sent == COMM.allreduce(received) and total_sent > 0
    assert store.length == length_before + received  # dead leavers stay until compact
    check_state(store, deco, ids_before)

    # everything is home now: a second migration sends nothing but still runs
    assert migrate(store, deco) == (0, 0)

    # push markers around a periodic box for a while and migrate each step
    deco = BoxDecomposition(COMM, [0.0, 0.0], [1.0, 1.0], PROC_SIZES, periodic=True)
    rng = np.random.default_rng(100 + RANK)
    for _ in range(15):
        count = store.length
        step = rng.uniform(-0.3, 0.3, size=(2, count))
        store.positions[:, :count] += xp.asarray(step)
        migrate(store, deco)
        store.compact_if_needed()
        check_state(store, deco, ids_before)
    store.compact()
    check_state(store, deco, ids_before)

    # markers every rank loaded alike: each keeps its share
    shared = ParticleStore(2, 3, id_allocator=IdAllocator(RANK, SIZE))
    points = np.random.default_rng(7).uniform(0.0, 1.0, size=(2, 500))
    shared.add(xp.asarray(points), xp.zeros((3, 500)))
    mpiparticles.keep_owned(shared, deco)
    assert global_count(shared, COMM) == 500

    # moments and phase-space densities: every rank gets the sums over all markers
    weights_all = gather_columns(shared.active_weights(), COMM, root=None)
    positions_all = gather_columns(shared.active_positions(), COMM, root=None)
    rng = np.random.default_rng(RANK)
    shared.set_active_velocities(xp.asarray(rng.normal(size=(3, shared.num_alive))))
    sums = velocity_moments(
        shared.active_velocities(), shared.active_weights(), 3, COMM
    )
    velocities = gather_columns(shared.active_velocities(), COMM, root=None)
    expected = velocity_moments(velocities, weights_all, 3)
    np.testing.assert_allclose(sums, expected, rtol=1e-12, atol=1e-12)
    axes = {"x": (0.0, 1.0, 4), "vy": (-2.0, 2.0, 6)}
    density = phase_space_density(
        shared.active_positions(),
        shared.active_velocities(),
        shared.active_weights(),
        axes,
        COMM,
    )
    np.testing.assert_allclose(
        density,
        phase_space_density(positions_all, velocities, weights_all, axes),
        rtol=1e-12,
    )

    if RANK == 0:
        print(f"OK size={SIZE} big={big} sent={total_sent}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        COMM.Abort(1)
