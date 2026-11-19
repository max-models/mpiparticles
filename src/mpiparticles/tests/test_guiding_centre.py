"""Guiding centres: the blocks in every backend, and the drifts and the mirror force.

The Pyccel blocks (``guiding_centre_pyccel.py``, run in Python) are the source of truth:
the NumPy blocks, the compiled kernels, the CUDA versions (on cunumpy's emulator, skipped
without a C++ compiler; end to end on the fake CuPy) and the Metal versions (on an
Apple-silicon Mac, float32) agree with them.
"""

import textwrap

import numpy as np
import pytest
from cunumpy.kernel_testing import (
    emulate_cuda_kernel,
    emulation_compiler,
    run_in_fake_cupy_subprocess,
)
from cunumpy.kernels import metal_available

import mpiparticles as mp
from mpiparticles.kernels.catalog import KERNELS, metal_kernel
from mpiparticles.kernels.guiding_centre import (
    guiding_centre_rates,
    guiding_centre_velocity,
)

N = 129
EMULATOR = pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
METAL = pytest.mark.skipif(
    not metal_available(), reason="MLX with a Metal GPU is not available"
)


def _rates_case(rng, n=N):
    e, b, g, k = (rng.normal(size=(3, n)) for _ in range(4))
    return [e, b, g, k, rng.normal(size=n), rng.uniform(0.1, 1.0, n), -1.3, 0.7]


def _velocity_case(rng, n=N):
    v, b_old, b_new = (rng.normal(size=(3, n)) for _ in range(3))
    return [v, b_old, b_new, rng.normal(size=n), rng.uniform(0.1, 1.0, n), 0.7]


def _pyccel_rates(e, b, g, k, v_par, mu, charge, mass):
    from mpiparticles.kernels.guiding_centre_pyccel import gc_rates

    rows = [
        gc_rates(*e[:, i], *b[:, i], *g[:, i], *k[:, i], v_par[i], mu[i], charge, mass)
        for i in range(v_par.shape[0])
    ]
    rows = np.array(rows).T
    return rows[:3], rows[3]


def _pyccel_velocity(v, b_old, b_new, v_par, mu, mass):
    from mpiparticles.kernels.guiding_centre_pyccel import gc_velocity

    rows = [
        gc_velocity(*v[:, i], *b_old[:, i], *b_new[:, i], v_par[i], mu[i], mass)
        for i in range(v_par.shape[0])
    ]
    return np.array(rows).T


def test_the_numpy_blocks_are_the_pyccel_blocks():
    rng = np.random.default_rng(0)
    args = _rates_case(rng)
    for got, expected in zip(
        guiding_centre_rates(*args), _pyccel_rates(*args), strict=True
    ):
        np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)
    args = _velocity_case(rng)
    np.testing.assert_allclose(
        guiding_centre_velocity(*args), _pyccel_velocity(*args), rtol=1e-12, atol=1e-12
    )


def test_the_compiled_kernels_are_the_pyccel_blocks():
    rng = np.random.default_rng(1)
    args = _rates_case(rng)
    drift, accel = np.empty((3, N)), np.empty(N)
    KERNELS["guiding_centre_rates"](*args, drift, accel)
    expected = _pyccel_rates(*args)
    np.testing.assert_allclose(drift, expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(accel, expected[1], rtol=1e-13, atol=1e-13)
    args = _velocity_case(rng)
    expected = _pyccel_velocity(*args)
    KERNELS["guiding_centre_velocity"](*args)
    np.testing.assert_allclose(args[0], expected, rtol=1e-12, atol=1e-12)


@EMULATOR
def test_the_cuda_kernels_are_the_pyccel_blocks():
    rng = np.random.default_rng(2)
    args = _rates_case(rng)
    drift, accel = np.empty((3, N)), np.empty(N)
    emulate_cuda_kernel(
        KERNELS["guiding_centre_rates"].cuda_kernel, *args, drift, accel, n_threads=N
    )
    expected = _pyccel_rates(*args)
    np.testing.assert_allclose(drift, expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(accel, expected[1], rtol=1e-13, atol=1e-13)
    args = _velocity_case(rng)
    expected = _pyccel_velocity(*args)
    emulate_cuda_kernel(
        KERNELS["guiding_centre_velocity"].cuda_kernel, *args, n_threads=N
    )
    np.testing.assert_allclose(args[0], expected, rtol=1e-12, atol=1e-12)


@EMULATOR
def test_the_step_runs_on_cupy_arrays():
    code = textwrap.dedent(
        """
        import numpy as np
        from cunumpy.kernel_testing import fake_cupy_session

        import cupy  # the fake CuPy, installed by importing cunumpy
        from mpiparticles import guiding_centre_step

        def fields(module):
            def at(points):
                n = points.shape[1]
                b = module.zeros((3, n)); b[2] = 1.0 + 0.1 * points[0]
                e = module.zeros((3, n)); e[0] = 0.3
                g = module.zeros((3, n)); g[0] = 0.1
                return e, b, g, module.zeros((3, n))
            return at

        rng = np.random.default_rng(3)
        x0, v0 = rng.normal(size=(2, 30)), rng.normal(size=(3, 30))
        x, v = x0.copy(), v0.copy()
        guiding_centre_step(x, v, fields(np), 0.1, -1.0, 2.0)
        with fake_cupy_session():
            xc, vc = cupy.array(x0, copy=True), cupy.array(v0, copy=True)
            guiding_centre_step(xc, vc, fields(cupy), 0.1, -1.0, 2.0)
            assert isinstance(xc, cupy.ndarray)
            np.testing.assert_allclose(cupy.asnumpy(xc), x, rtol=0, atol=1e-13)
            np.testing.assert_allclose(cupy.asnumpy(vc), v, rtol=0, atol=1e-13)
        """
    )
    run_in_fake_cupy_subprocess(code, timeout=600)


def _close(gpu, cpu, tol=3e-5):
    scale = max(float(np.abs(cpu).max()), 1e-30)
    np.testing.assert_allclose(gpu / scale, cpu / scale, rtol=0.0, atol=tol)


@METAL
def test_the_metal_kernels_agree():
    rng = np.random.default_rng(4)
    args = _rates_case(rng)
    expected = (np.empty((3, N)), np.empty(N))
    got = (np.empty((3, N)), np.empty(N))
    KERNELS["guiding_centre_rates"](*args, *expected)
    metal_kernel("guiding_centre_rates")(*args, *got)
    for a, b in zip(got, expected, strict=True):
        _close(a, b)
    args = _velocity_case(rng)
    expected = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    KERNELS["guiding_centre_velocity"](*expected)
    metal_kernel("guiding_centre_velocity")(*args)
    _close(args[0], expected[0])


# --- the motion ------------------------------------------------------------- #


def _uniform(e_vector, b_vector):
    def fields(points):
        n = points.shape[1]
        e = np.tile(np.asarray(e_vector, dtype=float)[:, None], n)
        b = np.tile(np.asarray(b_vector, dtype=float)[:, None], n)
        return e, b, np.zeros((3, n)), np.zeros((3, n))

    return fields


def test_e_cross_b_drift_and_parallel_streaming_are_exact():
    x = np.zeros((3, 4))
    v = np.array([[0.0, 1.0, 2.0, 3.0], [2.0, 0.0, 0.0, 0.0], [0.5, -0.5, 1.0, 0.0]])
    speed = np.linalg.norm(v, axis=0)
    mp.guiding_centre_step(
        x, v, _uniform([0.0, 3.0, 0.0], [0.0, 0.0, 2.0]), 0.5, 1.0, 1.0
    )
    # E x B / B^2 = (3 * 2, 0, 0) / 4 = 1.5 along x; v_z streams along B
    np.testing.assert_allclose(x[0], 0.75)
    np.testing.assert_allclose(x[2], 0.5 * np.array([0.5, -0.5, 1.0, 0.0]))
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), speed)


def test_grad_b_and_curvature_drifts():
    """B_z = B0 (1 + a x) gives the grad-B drift; circular field lines the curvature drift."""
    b0, a, q, m = 2.0, 0.1, -1.5, 3.0

    def graded(points):
        n = points.shape[1]
        b = np.zeros((3, n))
        b[2] = b0 * (1 + a * points[0])
        g = np.zeros((3, n))
        g[0] = b0 * a
        return np.zeros((3, n)), b, g, np.zeros((3, n))

    x, v = np.zeros((2, 1)), np.array([[0.4], [0.0], [0.0]])
    mu = mp.magnetic_moment(v, graded(x)[1], m)
    mp.guiding_centre_step(x, v, graded, 1e-3, q, m)
    # (mu / (q B)) (b x grad B) = (mu / (q B)) (z x x) dB/dx: along +y
    np.testing.assert_allclose(x[1] / 1e-3, mu / (q * b0) * b0 * a, rtol=1e-3)

    radius, b_phi = 5.0, 1.0

    def circular(points):  # B = b_phi phi_hat about the z axis, in the plane z = 0
        n = points.shape[1]
        r = np.hypot(points[0], points[1])
        b = np.stack([-points[1] / r, points[0] / r, np.zeros(n)]) * b_phi
        kappa = -np.stack([points[0], points[1], np.zeros(n)]) / r**2
        return np.zeros((3, n)), b, np.zeros((3, n)), kappa

    x, v = np.array([[radius], [0.0], [0.0]]), np.array([[0.0], [0.2], [0.0]])
    mp.guiding_centre_step(x, v, circular, 1e-4, q, m)
    # (m v_par^2 / (q B)) (b x kappa) = (m v_par^2 / (q B R)) (y x -x) = ... along +z
    np.testing.assert_allclose(
        x[2] / 1e-4, m * 0.2**2 / (q * b_phi * radius), rtol=1e-3
    )


def _bounce(dt, steps):
    def mirror(points):  # B_x = 1 + x^2, along x
        n = points.shape[1]
        b = np.zeros((3, n))
        b[0] = 1.0 + points[0] ** 2
        g = np.zeros((3, n))
        g[0] = 2.0 * points[0]
        return np.zeros((3, n)), b, g, np.zeros((3, n))

    store = mp.ParticleStore(ndim=1, nvel=3)
    store.add(np.array([[0.0, 0.3]]), np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))
    store.alive[1] = False  # a dead marker does not move
    positions, energies = [], []
    for _ in range(steps):
        mp.guiding_centre_push(store, mirror, dt, 1.0, 1.0)
        positions.append(float(store.positions[0, 0]))
        energies.append(0.5 * float(np.sum(store.velocities[:, 0] ** 2)))
    assert store.positions[0, 1] == 0.3
    mu = mp.magnetic_moment(
        store.velocities[:, :1], mirror(store.positions[:, :1])[1], 1.0
    )
    return np.array(positions), np.array(energies), float(mu[0])


def test_a_mirror_reflects_where_mu_b_is_the_energy_and_keeps_mu():
    positions, energies, mu = _bounce(0.01, 1000)
    # energy 1 = mu B(x) with mu = 0.5: B = 2, x_turn = 1
    assert positions.max() == pytest.approx(1.0, rel=1e-4)
    assert positions.min() == pytest.approx(-1.0, rel=1e-4)
    assert mu == pytest.approx(0.5, rel=1e-12)
    np.testing.assert_allclose(energies, 1.0, rtol=1e-4)


def test_the_step_is_second_order():
    errors = [np.abs(_bounce(dt, int(2.0 / dt))[1] - 1.0).max() for dt in (0.02, 0.01)]
    # second order at least (the energy error of this symmetric bounce falls as dt^3)
    assert errors[0] / errors[1] > 3.5


def test_a_store_needs_three_velocity_components():
    with pytest.raises(ValueError, match="three velocity components"):
        mp.guiding_centre_push(
            mp.ParticleStore(ndim=1, nvel=1),
            _uniform([0, 0, 0], [0, 0, 1]),
            0.1,
            1.0,
            1.0,
        )


def test_the_numpy_kernels_are_the_pyccel_blocks():
    from mpiparticles.kernels.guiding_centre_rates.guiding_centre_rates_numpy import (
        guiding_centre_rates as rates_kernel,
    )
    from mpiparticles.kernels.guiding_centre_velocity.guiding_centre_velocity_numpy import (
        guiding_centre_velocity as velocity_kernel,
    )

    rng = np.random.default_rng(5)
    args = _rates_case(rng)
    drift, accel = np.empty((3, N)), np.empty(N)
    rates_kernel(*args, drift, accel)
    expected = _pyccel_rates(*args)
    np.testing.assert_allclose(drift, expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(accel, expected[1], rtol=1e-13, atol=1e-13)
    args = _velocity_case(rng)
    expected = _pyccel_velocity(*args)
    velocity_kernel(*args)
    np.testing.assert_allclose(args[0], expected, rtol=1e-12, atol=1e-12)


def test_the_pyccel_blocks_are_what_the_generator_writes():
    from pathlib import Path

    from mpiparticles.kernels import (
        _generate_guiding_centre_pyccel,
        guiding_centre_pyccel,
    )

    committed = Path(guiding_centre_pyccel.__file__).read_text()
    assert _generate_guiding_centre_pyccel.source() == committed


def test_nothing_to_push():
    fields = _uniform([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="three velocity components"):
        mp.guiding_centre_step(
            np.zeros((1, 2)), np.zeros((2, 2)), fields, 0.1, 1.0, 1.0
        )
    empty = np.zeros((1, 0))
    mp.guiding_centre_step(empty, np.zeros((3, 0)), fields, 0.1, 1.0, 1.0)
    store = mp.ParticleStore(ndim=1, nvel=3)
    mp.guiding_centre_push(store, fields, 0.1, 1.0, 1.0)  # empty store
    store.add(np.array([[0.5]]), np.array([[1.0], [0.0], [0.0]]))
    store.alive[0] = False
    mp.guiding_centre_push(store, fields, 0.1, 1.0, 1.0)  # no alive marker
    assert store.positions[0, 0] == 0.5
