"""The Boris push: gyration, E×B drift, 1D3V to 3D3V, per-marker steps, against NumPy."""

import numpy as np
import pytest

from mpiparticles import ParticleStore, boris_push, boris_velocity
from mpiparticles.kernels.boris_velocity import kernel as velocity_kernel
from mpiparticles.kernels.boris_velocity.boris_velocity_numpy import (
    boris_velocity as boris_velocity_numpy,
)
from mpiparticles.kernels.push_boris import kernel
from mpiparticles.kernels.push_boris.push_boris_numpy import (
    push_boris as push_boris_numpy,
)


@pytest.fixture(autouse=True)
def compiled():
    """The tests run the compiled Pyccel kernel, never the NumPy fallback."""
    if not (
        kernel.host_kernel.kernel.available("pyccel")
        and velocity_kernel.host_kernel.kernel.available("pyccel")
    ):
        pytest.skip("Pyccel kernels unavailable")


def store_with(positions, velocities):
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    store = ParticleStore(ndim=positions.shape[0], nvel=3)
    store.add(positions, np.asarray(velocities, dtype=float))
    return store


def test_gyration_keeps_the_speed_and_turns_by_the_boris_angle():
    # a positive charge, q/m = 1, in B = 2 z: omega_c = 2, turning clockwise seen from +z
    store = store_with([[0.0], [0.0], [0.0]], [[1.0], [0.0], [0.5]])
    omega, dt, steps = 2.0, 0.05, 400
    angles, points = [], []
    for _ in range(steps):
        boris_push(store, [0.0, 0.0, 0.0], [0.0, 0.0, omega], dt, 1.0)
        vx, vy = store.velocities[0, 0], store.velocities[1, 0]
        angles.append(np.arctan2(vy, vx))
        points.append(store.positions[:2, 0].copy())
    v = store.velocities[:, 0]
    # the speed, and the velocity along B, are kept to rounding
    assert np.hypot(v[0], v[1]) == pytest.approx(1.0, rel=1e-13)
    assert v[2] == 0.5
    # each step turns the velocity by the Boris angle 2 arctan(omega dt / 2), not omega dt
    theta = 2 * np.arctan(omega * dt / 2)
    turn = -np.diff(np.unwrap(angles))
    np.testing.assert_allclose(turn, theta, rtol=1e-12)
    # the positions are the corners of a regular polygon: all on one circle, of radius
    # v_perp dt / (2 sin(theta / 2))
    points = np.array(points)
    a, b, c = points[0], points[7], points[19]
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    centre = (
        np.array(
            [
                np.dot(a, a) * (b[1] - c[1])
                + np.dot(b, b) * (c[1] - a[1])
                + np.dot(c, c) * (a[1] - b[1]),
                np.dot(a, a) * (c[0] - b[0])
                + np.dot(b, b) * (a[0] - c[0])
                + np.dot(c, c) * (b[0] - a[0]),
            ]
        )
        / d
    )
    radius = dt / (2 * np.sin(theta / 2))
    np.testing.assert_allclose(np.hypot(*(points - centre).T), radius, rtol=1e-10)
    # and drifts along B at the parallel speed
    assert store.positions[2, 0] == pytest.approx(0.5 * dt * steps, rel=1e-12)


def test_the_exb_drift_is_exact():
    # E = 3 y, B = 2 z: the drift E x B / B^2 = 1.5 x, for any charge and step
    e, b = np.array([0.0, 3.0, 0.0]), np.array([0.0, 0.0, 2.0])
    drift = np.cross(e, b) / b.dot(b)
    store = store_with(np.zeros((2, 1)), drift[:, None])
    for _ in range(100):
        boris_push(store, e, b, dt=0.3, charge_over_mass=-7.0)
    np.testing.assert_allclose(store.velocities[:, 0], drift, atol=1e-14)
    np.testing.assert_allclose(store.positions[:, 0], [1.5 * 30.0, 0.0], atol=1e-12)


def test_a_marker_at_rest_starts_to_gyrate_around_the_drift():
    # starting at rest, the mean velocity over whole gyrations is the E x B drift
    e, b = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    store = store_with(np.zeros((3, 1)), np.zeros((3, 1)))
    dt = 2 * np.pi / 2000
    for _ in range(2000 * 5):
        boris_push(store, e, b, dt, 1.0)
    mean_velocity = store.positions[:, 0] / (2000 * 5 * dt)
    np.testing.assert_allclose(mean_velocity, [1.0, 0.0, 0.0], atol=1e-3)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_compiled_kernel_matches_the_numpy_version(ndim):
    rng = np.random.default_rng(ndim)
    n = 257
    positions = rng.normal(size=(ndim, n))
    velocities = rng.normal(size=(3, n))
    e_field, b_field = rng.normal(size=(3, n)), rng.normal(size=(3, n))
    expected_positions, expected_velocities = positions.copy(), velocities.copy()
    push_boris_numpy(
        expected_positions, expected_velocities, e_field, b_field, -2.5, 0.1
    )
    store = store_with(positions, velocities)
    boris_push(store, e_field, b_field, 0.1, -2.5)
    np.testing.assert_allclose(store.velocities[:, :n], expected_velocities, rtol=1e-12)
    np.testing.assert_allclose(store.positions[:, :n], expected_positions, rtol=1e-12)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_positions_move_with_their_own_velocity_components(ndim):
    # a free marker: only the first ndim velocity components move the positions
    store = store_with(np.zeros((ndim, 1)), [[1.0], [2.0], [3.0]])
    boris_push(store, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5, 1.0)
    np.testing.assert_allclose(store.positions[:, 0], [0.5, 1.0, 1.5][:ndim])
    np.testing.assert_allclose(store.velocities[:, 0], [1.0, 2.0, 3.0])


def test_dead_markers_and_spare_capacity():
    store = store_with(np.zeros((2, 3)), np.ones((3, 3)))
    store.deactivate(np.array([False, True, False]))
    store.reserve(100)  # spare columns past length are not touched
    boris_push(store, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 1.0)
    np.testing.assert_allclose(store.positions[:, :3], np.ones((2, 3)))
    assert not store.alive[1]
    assert np.all(store.positions[:, 3:] == 0.0)


def test_an_empty_store_and_wrong_arguments():
    empty = ParticleStore(ndim=2, nvel=3)
    boris_push(empty, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.1, 1.0)
    with pytest.raises(ValueError, match="three velocity components"):
        boris_push(ParticleStore(ndim=1, nvel=1), [0, 0, 0], [0, 0, 1], 0.1, 1.0)
    store = store_with(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match=r"e_field must be \(3,\) or \(3, 2\)"):
        boris_push(store, np.zeros((3, 5)), [0.0, 0.0, 1.0], 0.1, 1.0)
    with pytest.raises(ValueError, match="b_field"):
        boris_push(store, [0.0, 0.0, 0.0], [0.0, 1.0], 0.1, 1.0)


def test_the_compiled_velocity_kernel_matches_the_numpy_version():
    rng = np.random.default_rng(7)
    n = 129
    velocities = rng.normal(size=(3, n))
    e_field, b_field = rng.normal(size=(3, n)), rng.normal(size=(3, n))
    qm, dt = rng.normal(size=n), rng.uniform(0.01, 0.2, size=n)
    expected = velocities.copy()
    boris_velocity_numpy(expected, e_field, b_field, qm, dt)
    boris_velocity(velocities, e_field, b_field, dt, qm)
    np.testing.assert_allclose(velocities, expected, rtol=1e-12)


def test_per_marker_values_are_the_same_as_one_push_per_marker():
    # three markers, each with its own step and charge, against three separate pushes
    rng = np.random.default_rng(3)
    positions, velocities = rng.normal(size=(2, 3)), rng.normal(size=(3, 3))
    e_field, b_field = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    dt, qm = np.array([0.1, 0.05, 0.2]), np.array([1.0, -3.0, 0.5])
    store = store_with(positions, velocities)
    store.reserve(10)  # the stored columns are a strided view of the arrays
    boris_push(store, e_field, b_field, dt, qm)
    for k in range(3):
        single = store_with(positions[:, k : k + 1], velocities[:, k : k + 1])
        boris_push(single, e_field[:, k], b_field[:, k], dt[k], qm[k])
        np.testing.assert_allclose(store.velocities[:, k], single.velocities[:, 0])
        np.testing.assert_allclose(store.positions[:, k], single.positions[:, 0])


def test_a_scalar_and_a_per_marker_array_mix():
    store = store_with(np.zeros((3, 2)), np.ones((3, 2)))
    reference = store_with(np.zeros((3, 2)), np.ones((3, 2)))
    boris_push(store, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], np.full(2, 0.1), 2.0)
    boris_push(reference, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.1, 2.0)
    np.testing.assert_allclose(store.velocities[:, :2], reference.velocities[:, :2])
    np.testing.assert_allclose(store.positions[:, :2], reference.positions[:, :2])


def test_the_velocity_update_leaves_the_positions_and_steps_back():
    # a negative step undoes a positive one (the Boris map is time reversible)
    rng = np.random.default_rng(11)
    start = rng.normal(size=(3, 5))
    e_field, b_field = rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
    qm, dt = rng.normal(size=5), np.full(5, 0.1)
    velocities = start.copy()
    boris_velocity(velocities, e_field, b_field, dt, qm)
    assert not np.allclose(velocities, start)
    boris_velocity(velocities, e_field, b_field, -dt, qm)
    np.testing.assert_allclose(velocities, start, rtol=1e-12, atol=1e-14)


def test_the_velocity_update_on_a_strided_view_and_no_markers():
    big = np.ones((3, 8))
    view = big[:, ::2]
    boris_velocity(view, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 1.0)
    np.testing.assert_allclose(big, 1.0)
    boris_velocity(np.zeros((3, 0)), [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.1, 1.0)


def test_wrong_per_marker_arguments():
    store = store_with(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match=r"dt must be a number or \(2,\)"):
        boris_push(store, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.ones(3), 1.0)
    with pytest.raises(ValueError, match="charge_over_mass"):
        boris_push(store, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.1, np.ones((2, 1)))
    with pytest.raises(ValueError, match=r"\(3, N\) array"):
        boris_velocity(np.zeros((2, 4)), [0, 0, 0], [0, 0, 1], 0.1, 1.0)
    with pytest.raises(ValueError, match="float64"):
        boris_velocity(
            np.zeros((3, 4), dtype=np.float32), [0, 0, 0], [0, 0, 1], 0.1, 1.0
        )
