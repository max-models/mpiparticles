import textwrap

import cunumpy as xp
import numpy as np
import pytest

from mpiparticles import IdAllocator, ParticleStore
from mpiparticles.tests.conftest import filled_store


def host(array):
    return np.asarray(xp.to_numpy(array))


def test_empty_store_layout():
    store = ParticleStore(2, 3, capacity=4)
    assert store.positions.shape == (2, 4)
    assert store.velocities.shape == (3, 4)
    assert store.weights.shape == (4,)
    assert store.alive.shape == (4,) and store.alive.dtype == bool
    assert store.ids.dtype == np.int64
    assert (store.length, store.capacity, store.num_alive) == (0, 4, 0)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        ParticleStore(3, 2)
    with pytest.raises(ValueError):
        ParticleStore(2, 3, capacity=0)
    store = ParticleStore(2, 3)
    with pytest.raises(ValueError):
        store.add(xp.zeros((3, 1)), xp.zeros((3, 1)))
    with pytest.raises(ValueError):
        store.add(xp.zeros((2, 2)), xp.zeros((3, 1)))
    with pytest.raises(ValueError):
        store.add(xp.zeros((2, 2)), xp.zeros((3, 2)), weights=xp.ones(3))
    with pytest.raises(ValueError):
        store.add(xp.zeros((2, 2)), xp.zeros((3, 2)), ids=xp.asarray([1.5, 2.5]))


def test_add_and_growth_by_doubling():
    store = filled_store(5, capacity=4)
    assert store.capacity == 8 and store.length == 5
    store.add(xp.zeros((2, 20)), xp.zeros((3, 20)))
    assert store.capacity == 32 and store.length == 25
    # the data survived the reallocations
    assert host(store.positions[0, :5]).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert host(store.weights[:5]).tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert host(store.weights[5:25]).tolist() == [1.0] * 20


def test_reserve_rebinds_arrays_and_views_go_stale():
    store = filled_store(3, capacity=4)
    view = store.positions[:, :3]
    store.reserve(100)
    assert store.capacity >= 100
    store.positions[0, 0] = 99.0
    assert host(view)[0, 0] == 0.0  # the old view no longer follows the store
    store.reserve(10)  # no shrink
    assert store.capacity >= 100


def test_add_weights_scalar_and_default():
    store = ParticleStore(1, 1)
    store.add(xp.zeros((1, 2)), xp.zeros((1, 2)))
    store.add(xp.zeros((1, 2)), xp.zeros((1, 2)), weights=2.5)
    assert host(store.weights[:4]).tolist() == [1.0, 1.0, 2.5, 2.5]


def test_ids_unique_and_returned():
    store = ParticleStore(1, 1)
    first = store.add(xp.zeros((1, 3)), xp.zeros((1, 3)))
    second = store.add(xp.zeros((1, 2)), xp.zeros((1, 2)))
    assert host(first).tolist() == [0, 1, 2] and host(second).tolist() == [3, 4]
    assert len(set(host(store.ids[:5]).tolist())) == 5


def test_add_with_ids_moves_the_allocator_past_them():
    store = ParticleStore(1, 1, id_allocator=IdAllocator(1, 4))
    store.add(xp.zeros((1, 1)), xp.zeros((1, 1)), ids=xp.asarray([2**60 + 3]))
    new = store.add(xp.zeros((1, 2)), xp.zeros((1, 2)))
    assert all(int(i) % 4 == 1 and int(i) > 2**60 + 3 for i in host(new))


def test_deactivate_compact_keeps_order_and_ids():
    store = filled_store(6)
    ids = host(store.ids[:6]).copy()
    store.deactivate(xp.asarray([False, True, False, True, True, False]))
    assert store.length == 6 and store.num_alive == 3
    assert host(store.active_ids()).tolist() == ids[[0, 2, 5]].tolist()
    store.compact()
    assert store.length == 3 and store.num_alive == 3
    assert host(store.ids[:3]).tolist() == ids[[0, 2, 5]].tolist()
    assert host(store.weights[:3]).tolist() == [1.0, 3.0, 6.0]
    assert host(store.velocities[0, :3]).tolist() == [0.0, -2.0, -5.0]
    # the freed tail is reset
    assert not host(store.alive[3:6]).any() and host(store.ids[3:6]).tolist() == [0] * 3


def test_compact_if_needed_threshold():
    store = filled_store(8)
    store.deactivate(xp.asarray([True, True] + [False] * 6))
    assert not store.compact_if_needed(0.25)  # 2 of 8 is exactly the fraction
    store.deactivate(xp.asarray([False, False, True] + [False] * 5))
    assert store.compact_if_needed(0.25) and store.length == 5
    assert not ParticleStore(1, 1).compact_if_needed()


def test_remove_index():
    store = filled_store(4)
    store.remove(1)
    assert store.length == 3
    assert host(store.weights[:3]).tolist() == [1.0, 3.0, 4.0]
    assert host(store.ids[:3]).tolist() == [0, 2, 3]
    store.remove(2)
    assert store.length == 2 and host(store.ids[:3]).tolist() == [0, 2, 0]
    with pytest.raises(IndexError):
        store.remove(2)


def test_clear_keeps_the_id_counter():
    store = filled_store(3)
    store.clear()
    assert store.length == 0 and store.num_alive == 0
    new = store.add(xp.zeros((2, 1)), xp.zeros((3, 1)))
    assert host(new).tolist() == [3]


def test_sort_by_key_reorders_all_fields_and_drops_dead():
    store = filled_store(6)
    ids = host(store.ids[:6]).copy()
    store.deactivate(xp.asarray([False, False, True, False, False, False]))
    keys = xp.asarray([3, 1, 0, 3, 0, 2])  # the dead row has the smallest key
    store.sort_by_key(keys)
    assert store.length == 5 and store.num_alive == 5
    # stable: equal keys keep their order; the dead row is gone
    order = [4, 1, 5, 0, 3]
    assert host(store.ids[:5]).tolist() == ids[order].tolist()
    assert host(store.weights[:5]).tolist() == [float(i + 1) for i in order]
    assert host(store.positions[0, :5]).tolist() == [float(i) for i in order]
    assert host(store.velocities[1, :5]).tolist() == [-float(i) - 1 for i in order]


def _reference_sort(store, keys):
    """What sort_by_key must leave: the alive columns, stably ordered by key."""
    alive = host(store.alive[: store.length])
    order = np.flatnonzero(alive)[np.argsort(host(keys)[alive], kind="stable")]
    return (
        host(store.positions)[:, order],
        host(store.velocities)[:, order],
        host(store.weights)[order],
        host(store.ids)[order],
    )


@pytest.mark.parametrize(
    ("dtype", "dead_key"),
    [
        (np.int64, None),
        (np.int32, None),
        (np.uint16, None),
        (np.int64, np.iinfo(np.int64).max),  # no key above it: compacts first
        (np.uint8, np.iinfo(np.uint8).max),
        (np.float64, None),  # not integers: compacts first
    ],
)
@pytest.mark.parametrize("dead_fraction", [0.0, 0.3, 1.0])
def test_sort_by_key_matches_a_stable_sort_of_the_alive(dtype, dead_key, dead_fraction):
    rng = np.random.default_rng(1)
    n = 50
    store = filled_store(n, capacity=64)
    store.deactivate(xp.asarray(rng.random(n) < dead_fraction))
    keys = rng.integers(0, 6, n).astype(dtype)
    if dead_key is not None:
        keys[rng.integers(0, n, 5)] = dead_key  # the largest key, alive or dead
    expected = _reference_sort(store, keys)
    count = expected[2].size
    store.sort_by_key(xp.asarray(keys))
    assert (store.length, store.num_alive) == (count, count)
    assert np.array_equal(host(store.positions[:, :count]), expected[0])
    assert np.array_equal(host(store.velocities[:, :count]), expected[1])
    assert np.array_equal(host(store.weights[:count]), expected[2])
    assert np.array_equal(host(store.ids[:count]), expected[3])


@pytest.mark.parametrize("dead_fraction", [0.0, 0.3, 1.0])
def test_counting_sort_matches_a_stable_sort_of_the_alive(dead_fraction):
    rng = np.random.default_rng(2)
    n, num_keys = 50, 7
    store = filled_store(n, capacity=64)
    dead = rng.random(n) < dead_fraction
    store.deactivate(xp.asarray(dead))
    keys = rng.integers(0, num_keys, n)
    keys[dead] = rng.integers(-5, 50, int(dead.sum()))  # ignored
    expected = _reference_sort(store, keys)
    count = expected[2].size
    starts = store.sort_by_key(xp.asarray(keys), num_keys=num_keys)
    assert (store.length, store.num_alive) == (count, count)
    assert np.array_equal(host(store.positions[:, :count]), expected[0])
    assert np.array_equal(host(store.velocities[:, :count]), expected[1])
    assert np.array_equal(host(store.weights[:count]), expected[2])
    assert np.array_equal(host(store.ids[:count]), expected[3])
    counts = np.bincount(keys[~dead], minlength=num_keys)
    assert host(starts).tolist() == [0, *np.cumsum(counts).tolist()]


@pytest.mark.parametrize("key", [-1, 7])
def test_counting_sort_refuses_alive_keys_out_of_range(key):
    store = filled_store(4)
    with pytest.raises(ValueError, match="outside 0 .. 6"):
        store.sort_by_key(xp.asarray([0, key, 2, 3]), num_keys=7)
    with pytest.raises(ValueError, match="integers"):
        store.sort_by_key(xp.zeros(4), num_keys=7)


def test_repeated_sorts_swap_the_storage_and_keep_the_free_columns_empty():
    rng = np.random.default_rng(3)
    store = filled_store(40, capacity=64)
    for size in (30, 10, 25):
        store.deactivate(xp.asarray(rng.random(store.length) < 0.2))
        positions = store.positions
        keys = rng.integers(0, 5, store.length)
        expected = _reference_sort(store, keys)
        store.sort_by_key(xp.asarray(keys), num_keys=5)
        assert store.positions is not positions  # gathered into the second set
        count = store.length
        assert np.array_equal(host(store.ids[:count]), expected[3])
        # the columns beyond length are in the empty state
        assert not host(store.positions[:, count:]).any()
        assert not host(store.velocities[:, count:]).any()
        assert (host(store.weights[count:]) == 1.0).all()
        assert not host(store.ids[count:]).any()
        assert not host(store.alive[count:]).any()
        store.clear()
        store.add(rng.random((2, size)), rng.random((3, size)))


def test_the_counting_sort_kernel_versions_agree():
    from mpiparticles.kernels.counting_sort import counting_sort_numpy
    from mpiparticles.kernels.counting_sort import kernel as counting_sort

    rng = np.random.default_rng(4)
    keys = rng.integers(0, 9, 200)
    alive = rng.random(200) > 0.25
    results = []
    for version in (counting_sort, counting_sort_numpy.counting_sort):
        starts, order = np.empty(11, np.int64), np.empty(200, np.int64)
        version(keys, alive, starts, order)
        results.append((starts, order))
    assert np.array_equal(results[0][0], results[1][0])
    assert np.array_equal(results[0][1], results[1][1])
    keys[np.flatnonzero(alive)[0]] = 9
    for version in (counting_sort, counting_sort_numpy.counting_sort):
        starts = np.empty(11, np.int64)
        version(keys, alive, starts, np.empty(200, np.int64))
        assert starts[0] == -1


def test_sort_keeps_ids_exact_above_2_53():
    store = ParticleStore(1, 1)
    big = 2**62 + np.arange(4, dtype=np.int64) * 3 + 1
    store.add(xp.zeros((1, 4)), xp.zeros((1, 4)), ids=xp.asarray(big))
    store.sort_by_key(xp.asarray([3, 2, 1, 0]))
    assert host(store.ids[:4]).tolist() == big[::-1].tolist()


def test_sort_by_key_validates_shape():
    with pytest.raises(ValueError):
        filled_store(3).sort_by_key(xp.zeros(2, dtype=int))


def test_active_accessors_and_setters():
    store = filled_store(4)
    store.deactivate(xp.asarray([True, False, False, False]))
    assert store.active_positions().shape == (2, 3)
    assert store.active_velocities().shape == (3, 3)
    assert host(store.active_weights()).tolist() == [2.0, 3.0, 4.0]
    store.set_active_weights(xp.asarray([7.0, 8.0, 9.0]))
    store.set_active_positions(xp.zeros((2, 3)))
    store.set_active_velocities(xp.ones((3, 3)))
    assert host(store.weights[:4]).tolist() == [1.0, 7.0, 8.0, 9.0]  # row 0 is dead
    assert host(store.positions[:, :4])[:, 0].tolist() == [
        0.0,
        0.5,
    ]  # the dead row is untouched
    assert host(store.positions[:, 1:4]).sum() == 0.0
    assert host(store.velocities[:, 1:4]).min() == 1.0


def test_snapshot_restore_round_trip():
    store = filled_store(6, capacity=4)
    store.deactivate(xp.asarray([False, True, False, False, False, False]))
    snapshot = store.snapshot()
    expected = {
        name: host(getattr(store, name)[..., : store.length]).copy()
        for name in ("positions", "velocities", "weights", "alive", "ids")
    }
    # change everything, including growth and the id counter
    store.compact()
    store.add(xp.ones((2, 40)), xp.ones((3, 40)))
    store.positions[:, :3] = -1.0
    store.restore(snapshot)
    assert store.length == 6 and store.num_alive == 5
    for name, value in expected.items():
        assert np.array_equal(host(getattr(store, name)[..., :6]), value), name
    assert not host(store.alive[6:]).any()
    assert store.id_allocator.next_id == snapshot.next_id == 6
    # a restored store keeps allocating unique ids
    new = store.add(xp.zeros((2, 1)), xp.zeros((3, 1)))
    assert host(new).tolist() == [6]
    # the snapshot is a copy
    store.positions[:, :6] = 55.0
    assert host(snapshot.positions).max() < 55.0


def test_restore_into_another_store_and_mismatch():
    store = filled_store(3)
    other = ParticleStore(2, 3, capacity=1)
    other.restore(store.snapshot())
    assert other.num_alive == 3 and other.capacity >= 3
    with pytest.raises(ValueError):
        ParticleStore(1, 3).restore(store.snapshot())


def test_marker_set_view_is_a_non_copy_transpose():
    store = filled_store(4)
    view = store.as_marker_set()
    assert view.positions.shape == (4, 2)
    assert view.velocities.shape == (4, 3)
    assert view.weights.shape == (4,)
    view.velocities[:, 0] *= 2.0
    view.velocities[1] = 7.0
    assert host(store.velocities[0, :4]).tolist() == [0.0, 7.0, -4.0, -6.0]
    assert host(store.velocities[:, 1]).tolist() == [7.0, 7.0, 7.0]
    view.weights[0] = 10.0
    assert host(store.weights[0]) == 10.0
    assert np.shares_memory(host(view.positions), host(store.positions))


def test_marker_set_remove_and_add():
    store = filled_store(5)
    view = store.as_marker_set()
    view.remove(xp.asarray([True, False, True, False, False]))
    assert store.length == 3 and view.positions.shape == (3, 2)
    assert host(view.weights).tolist() == [2.0, 4.0, 5.0]
    view.add(xp.ones((2, 2)), xp.zeros((2, 3)), xp.asarray([8.0, 9.0]))
    assert store.num_alive == 5 and view.velocities.shape == (5, 3)
    assert host(view.weights).tolist() == [2.0, 4.0, 5.0, 8.0, 9.0]
    assert host(store.ids[:5]).tolist() == [1, 3, 4, 5, 6]


def test_marker_set_compacts_dead_rows():
    store = filled_store(4)
    store.deactivate(xp.asarray([False, True, False, False]))
    view = store.as_marker_set()
    assert store.length == 3 and view.weights.shape == (3,)
    store.deactivate(xp.asarray([True, False, False]))
    assert view.weights.shape == (2,)  # compacted again on access


def test_float32_store():
    store = ParticleStore(1, 1, dtype=np.float32)
    store.add(xp.zeros((1, 2)), xp.zeros((1, 2)))
    assert store.positions.dtype == np.float32 and store.ids.dtype == np.int64


def test_replacing_the_id_allocator_never_reuses_a_stored_id():
    store = ParticleStore(1, 1)
    store.add(np.zeros((1, 3)), np.zeros((1, 3)))
    assert store.ids.tolist()[:3] == [0, 1, 2]

    store.id_allocator = IdAllocator(rank=1, size=4)
    new = store.add(np.zeros((1, 2)), np.zeros((1, 2)))

    assert set(new.tolist()).isdisjoint({0, 1, 2})
    assert all(int(i) % 4 == 1 for i in new)


def test_both_sorts_run_on_cupy_arrays():
    from cunumpy.kernel_testing import run_in_fake_cupy_subprocess

    code = textwrap.dedent(
        """
        import numpy as np
        from cunumpy.kernel_testing import fake_cupy_session

        import cupy  # the fake CuPy, installed by importing cunumpy
        from mpiparticles import ParticleStore

        rng = np.random.default_rng(5)
        n = 30
        x, v = rng.random((2, n)), rng.random((3, n))
        keys = rng.integers(0, 4, n)
        dead = rng.random(n) < 0.3
        alive = ~dead
        order = np.flatnonzero(alive)[np.argsort(keys[alive], kind="stable")]
        with fake_cupy_session():
            for num_keys in (None, 4):
                store = ParticleStore(2, 3, capacity=40)
                store.add(cupy.array(x), cupy.array(v))
                store.deactivate(cupy.array(dead))
                starts = store.sort_by_key(cupy.array(keys), num_keys=num_keys)
                assert isinstance(store.positions, cupy.ndarray)
                count = store.length
                assert count == alive.sum()
                np.testing.assert_array_equal(
                    cupy.asnumpy(store.positions[:, :count]), x[:, order]
                )
                np.testing.assert_array_equal(
                    cupy.asnumpy(store.velocities[:, :count]), v[:, order]
                )
                if num_keys is not None:
                    counts = np.bincount(keys[alive], minlength=4)
                    assert list(starts) == [0, *np.cumsum(counts)]
            try:
                store.sort_by_key(cupy.array(np.full(store.length, 9)), num_keys=4)
            except ValueError as error:
                assert "outside" in str(error)
            else:
                raise AssertionError("an alive key out of range was accepted")
        """
    )
    run_in_fake_cupy_subprocess(code, timeout=600)
