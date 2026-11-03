"""Serial runs make no MPI calls."""

import doctest

import cunumpy as xp
import numpy as np
import pytest

import mpiparticles
from mpiparticles import (
    BoxDecomposition,
    gather_columns,
    global_count,
    keep_owned,
    migrate,
)
from mpiparticles.tests.conftest import FakeComm, filled_store


class NoCommunication(FakeComm):
    def __getattr__(self, name):
        raise AssertionError(f"MPI call {name} in a serial run")


def test_serial_migrate_makes_no_mpi_call():
    store = filled_store(5)
    deco = BoxDecomposition(NoCommunication(0, 1), [0, 0], [1, 1], [1, 1])
    # a position far outside the box has an owner too: this rank
    assert migrate(store, deco) == (0, 0)
    assert store.num_alive == 5


def test_serial_helpers():
    comm = NoCommunication(0, 1)
    store = filled_store(5)
    assert global_count(store, comm) == global_count(store) == 5
    ids = gather_columns(store.active_ids(), comm)
    assert ids.tolist() == [0, 1, 2, 3, 4]
    assert gather_columns(store.active_positions()).shape == (2, 5)


def test_keep_owned_drops_foreign_markers():
    store = filled_store(10, ndim=1, nvel=1)  # positions 0..9
    deco = BoxDecomposition(FakeComm(1, 2), [0.0], [10.0], [2])
    assert keep_owned(store, deco) == 5
    assert store.length == 5
    assert np.asarray(xp.to_numpy(store.positions[0, :5])).tolist() == [5, 6, 7, 8, 9]
    assert np.asarray(xp.to_numpy(store.ids[:5])).tolist() == [5, 6, 7, 8, 9]
    assert keep_owned(store, deco) == 0


def test_the_empty_store_is_fine():
    store = mpiparticles.ParticleStore(2, 2)
    deco = BoxDecomposition(FakeComm(0, 2), [0, 0], [1, 1], [2, 1])
    assert keep_owned(store, deco) == 0


@pytest.mark.parametrize("module", ["ids", "store", "decomposition", "moments"])
def test_doctests(module):
    mod = __import__(f"mpiparticles.{module}", fromlist=["x"])
    assert doctest.testmod(mod, extraglobs={"xp": xp}).failed == 0
