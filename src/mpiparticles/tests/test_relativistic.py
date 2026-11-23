"""The relativistic Boris push and the proper-velocity helpers.

Gyration at omega_c / gamma with |u| kept, hyperbolic motion in a uniform electric field
(second order), the relativistic E×B drift, the non-relativistic limit, per-marker values,
and the compiled kernels against their NumPy versions.
"""

import numpy as np
import pytest

from mpiparticles import (
    SPEED_OF_LIGHT,
    ParticleStore,
    boris_push,
    boris_velocity,
    coordinate_velocity,
    kinetic_energy,
    lorentz_factor,
    proper_velocity,
)
from mpiparticles.kernels.push_relativistic_boris import kernel as push_kernel
from mpiparticles.kernels.push_relativistic_boris.push_relativistic_boris_numpy import (
    push_relativistic_boris as push_numpy,
)
from mpiparticles.kernels.relativistic_boris_velocity import kernel as velocity_kernel
from mpiparticles.kernels.relativistic_boris_velocity.relativistic_boris_velocity_numpy import (
    relativistic_boris_velocity as velocity_numpy,
)

C = 1.0  # normalised units: c = 1, q/m = 1


@pytest.fixture(autouse=True)
def compiled():
    """The tests run the compiled Pyccel kernels, never the NumPy fallback."""
    if not (
        push_kernel.host_kernel.kernel.available("pyccel")
        and velocity_kernel.host_kernel.kernel.available("pyccel")
    ):
        pytest.skip("Pyccel kernels unavailable")


def store_with(positions, velocities):
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    store = ParticleStore(ndim=positions.shape[0], nvel=3)
    store.add(positions, np.asarray(velocities, dtype=float))
    return store


def push(store, e, b, dt, qm=1.0, c=C):
    boris_push(store, e, b, dt, qm, relativistic=True, speed_of_light=c)


def test_proper_velocities_lorentz_factors_and_energies():
    v = np.array([[0.6, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.0]])
    u = proper_velocity(v, C)
    np.testing.assert_allclose(lorentz_factor(u, C), [1.25, 5 / 3, 1.0])
    np.testing.assert_allclose(u[:, 0], [0.75, 0.0, 0.0])
    np.testing.assert_allclose(coordinate_velocity(u, C), v, atol=1e-15)
    np.testing.assert_allclose(
        kinetic_energy(u, 2.0, C), 2.0 * (lorentz_factor(u, C) - 1)
    )
    # slow markers: the classical energy, to full precision
    slow = np.array([[3.0], [4.0], [0.0]])
    assert kinetic_energy(slow, 2.0)[0] == pytest.approx(25.0, rel=1e-15)
    with pytest.raises(ValueError, match="slower than light"):
        proper_velocity(np.array([[1.0], [0.0], [0.0]]), C)


def test_gyration_keeps_u_and_turns_at_the_relativistic_frequency():
    gamma = 5.0
    u0 = np.sqrt(gamma**2 - 1.0)  # |u| = gamma |v| with c = 1
    store = store_with([[0.0], [0.0]], [[u0], [0.0], [0.0]])
    b, dt = 1.0, 0.1
    push(store, [0.0, 0.0, 0.0], [0.0, 0.0, b], dt)
    u = store.velocities[:, 0]
    # one Boris rotation by 2 arctan(omega_c dt / (2 gamma)), clockwise for q > 0
    angle = 2 * np.arctan(b * dt / (2 * gamma))
    np.testing.assert_allclose(
        u, [u0 * np.cos(angle), -u0 * np.sin(angle), 0.0], atol=1e-14
    )
    x = []
    for _ in range(10_000):
        push(store, [0.0, 0.0, 0.0], [0.0, 0.0, b], dt)
        x.append(store.positions[0, 0])
    assert np.linalg.norm(store.velocities[:, 0]) == pytest.approx(u0, rel=1e-12)
    # the orbit's diameter is twice the relativistic gyroradius |u| / omega_c
    assert (max(x) - min(x)) / 2 == pytest.approx(u0 / b, rel=1e-3)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_hyperbolic_motion_is_second_order(ndim):
    # a uniform electric field from rest: u = a t exactly, x = (sqrt(1 + (a t)^2) - 1) / a
    a, t_end = 2.0, 3.0  # gamma = sqrt(37) at the end

    def error(steps):
        dt = t_end / steps
        positions = np.zeros((ndim, 1))
        store = store_with(positions, np.zeros((3, 1)))
        e = np.zeros(3)
        e[ndim - 1] = a
        boris_velocity(
            store.velocities[:, :1], e, [0.0, 0.0, 0.0], -0.5 * dt, 1.0,
            relativistic=True, speed_of_light=C,
        )  # fmt: skip
        for _ in range(steps):
            push(store, e, [0.0, 0.0, 0.0], dt)
        assert store.velocities[ndim - 1, 0] == pytest.approx(
            a * (t_end - dt / 2), rel=1e-13
        )
        assert np.abs(coordinate_velocity(store.velocities[:, :1], C)).max() < C
        exact = (np.sqrt(1 + (a * t_end) ** 2) - 1) / a
        return abs(store.positions[ndim - 1, 0] - exact)

    coarse, fine = error(50), error(100)
    assert coarse < 2e-3
    assert coarse / fine == pytest.approx(4.0, rel=0.05)


def test_the_relativistic_exb_drift():
    # E = 0.6 c B: the guiding centre drifts at E / B = 0.6 c, the gyration at gamma ~ 2.8
    e, b = 0.6, 1.0
    store = store_with([[0.0], [0.0]], np.zeros((3, 1)))
    dt, steps = 0.02, 20_000
    for _ in range(steps):
        push(store, [0.0, e, 0.0], [0.0, 0.0, b], dt)
    assert store.positions[0, 0] / (steps * dt) == pytest.approx(e / b, rel=2e-3)


def test_slow_markers_follow_the_classical_boris_push():
    rng = np.random.default_rng(0)
    x, v, e, b = rng.normal(size=(4, 3, 20))
    classical = store_with(x, v)
    boris_push(classical, e, b, 0.1, -2.0)
    relativistic = store_with(x, v)
    boris_push(relativistic, e, b, 0.1, -2.0, relativistic=True)  # c = 3e8 >> |v| ~ 1
    for got, want in (
        (relativistic.positions, classical.positions),
        (relativistic.velocities, classical.velocities),
    ):
        np.testing.assert_allclose(got[:, :20], want[:, :20], rtol=1e-12, atol=1e-14)


def test_markers_stay_slower_than_light_in_strong_fields():
    rng = np.random.default_rng(1)
    store = store_with(np.zeros((3, 50)), rng.normal(size=(3, 50)))
    for _ in range(20):
        push(store, rng.normal(0, 100, (3, 50)), rng.normal(0, 100, (3, 50)), 1.0)
    speed = np.linalg.norm(coordinate_velocity(store.velocities[:, :50], C), axis=0)
    assert speed.max() < C
    assert lorentz_factor(store.velocities[:, :50], C).min() > 10.0


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_compiled_kernels_match_the_numpy_versions(ndim):
    rng = np.random.default_rng(ndim)
    x, u = rng.normal(size=(ndim, 30)), rng.normal(size=(3, 30)) * SPEED_OF_LIGHT
    e, b = rng.normal(size=(3, 30)) * 1e6, rng.normal(size=(3, 30))
    expected = [x.copy(), u.copy()]
    push_numpy(*expected, e, b, -1.7e11, 1e-12, SPEED_OF_LIGHT)
    store = store_with(x, u)
    boris_push(store, e, b, 1e-12, -1.7e11, relativistic=True)
    np.testing.assert_allclose(store.positions[:, :30], expected[0], rtol=1e-13)
    np.testing.assert_allclose(store.velocities[:, :30], expected[1], rtol=1e-13)

    qm, dt = rng.normal(size=30) * 1e11, rng.uniform(1e-13, 1e-12, 30)
    expected = u.copy()
    velocity_numpy(expected, e, b, qm, dt, SPEED_OF_LIGHT)
    got = u.copy()
    boris_velocity(got, e, b, dt, qm, relativistic=True)
    np.testing.assert_allclose(got, expected, rtol=1e-13)


def test_per_marker_values_are_the_same_as_one_push_per_marker():
    rng = np.random.default_rng(2)
    x, u, e, b = rng.normal(size=(4, 3, 6))
    dt, qm = rng.uniform(0.05, 0.5, 6), rng.normal(size=6)
    store = store_with(x, u)
    push(store, e, b, dt, qm)
    for k in range(6):
        single = store_with(x[:, k : k + 1], u[:, k : k + 1])
        push(single, e[:, k], b[:, k], dt[k], qm[k])
        np.testing.assert_allclose(
            store.positions[:, k], single.positions[:, 0], rtol=1e-14
        )
        np.testing.assert_allclose(
            store.velocities[:, k], single.velocities[:, 0], rtol=1e-14
        )


def test_the_velocity_update_steps_back_and_takes_strided_views():
    rng = np.random.default_rng(3)
    u0, e, b = rng.normal(size=(3, 3, 8))
    u = u0.copy()
    boris_velocity(u, e, b, 0.2, 1.5, relativistic=True, speed_of_light=C)
    boris_velocity(u, e, b, -0.2, 1.5, relativistic=True, speed_of_light=C)
    # time-reversible: the rotation uses gamma after the first half kick in both directions
    np.testing.assert_allclose(u, u0, rtol=1e-12, atol=1e-13)
    columns = np.asfortranarray(u0.copy())
    expected = u0.copy()
    boris_velocity(expected, e, b, 0.2, 1.5, relativistic=True, speed_of_light=C)
    boris_velocity(columns, e, b, 0.2, 1.5, relativistic=True, speed_of_light=C)
    np.testing.assert_allclose(columns, expected, rtol=1e-14)
