"""The kernels' CUDA and Metal versions against the CPU.

The CUDA versions run on the CPU through cunumpy's emulator (skipped without a C++
compiler), and :func:`boris_velocity` and :func:`boris_push` run end to end on CuPy
arrays in a child process on cunumpy's fake CuPy. The Metal versions run on the GPU of an
Apple-silicon Mac (skipped without MLX and a Metal GPU); it computes in float32, so they
agree to about 1e-5 of the largest value.
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

from mpiparticles.kernels.catalog import (
    KERNEL_NAMES,
    KERNELS,
    METAL_KERNEL_NAMES,
    metal_kernel,
)

N = 257
EMULATOR = pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
METAL = pytest.mark.skipif(
    not metal_available(), reason="MLX with a Metal GPU is not available"
)


def _fields(rng, n=N):
    """Velocities, E and B at the markers, ``(3, n)`` each, in units where q/m dt ~ 1."""
    return rng.normal(size=(3, n)), rng.normal(size=(3, n)), rng.normal(size=(3, n))


def _velocity_case(rng):
    v, e, b = _fields(rng)
    return [v, e, b, rng.normal(size=N), rng.uniform(0.05, 0.5, N)]


def _push_case(rng, ndim):
    v, e, b = _fields(rng)
    return [rng.uniform(-1, 1, (ndim, N)) + 1000.0, v, e, b, -1.7, 0.3]


def _copies(args):
    return [a.copy() if isinstance(a, np.ndarray) else a for a in args]


def test_the_catalog_has_every_kernel():
    assert set(KERNELS) == set(KERNEL_NAMES) == set(METAL_KERNEL_NAMES)
    with pytest.raises(KeyError, match="no Metal version"):
        metal_kernel("push_nothing")


@pytest.mark.skipif(metal_available(), reason="a Metal GPU is available")
def test_metal_kernels_need_a_metal_gpu():
    with pytest.raises(RuntimeError, match="MLX and a Metal GPU"):
        metal_kernel("boris_velocity")


@EMULATOR
def test_the_cuda_velocity_kernel_agrees():
    args = _velocity_case(np.random.default_rng(0))
    expected = _copies(args)
    KERNELS["boris_velocity"].host_kernel.kernel.python(*expected)
    result = _copies(args)
    emulate_cuda_kernel(KERNELS["boris_velocity"].cuda_kernel, *result, n_threads=N)
    np.testing.assert_allclose(result[0], expected[0], rtol=1e-13, atol=1e-13)


@EMULATOR
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_cuda_push_kernel_agrees(ndim):
    args = _push_case(np.random.default_rng(ndim), ndim)
    expected = _copies(args)
    KERNELS["push_boris"].host_kernel.kernel.python(*expected)
    result = _copies(args)
    emulate_cuda_kernel(KERNELS["push_boris"].cuda_kernel, *result, n_threads=N)
    for i in (0, 1):
        np.testing.assert_allclose(result[i], expected[i], rtol=1e-13, atol=1e-13)


@EMULATOR
def test_the_push_runs_on_cupy_arrays():
    # the public functions on CuPy arrays: scalar and per-marker steps, (3,) fields and
    # transposed columns, all launched as CUDA kernels on the fake CuPy
    code = textwrap.dedent(
        """
        import numpy as np
        from cunumpy.kernel_testing import fake_cupy_session

        import cupy  # the fake CuPy, installed by importing cunumpy
        from mpiparticles import ParticleStore, boris_push, boris_velocity
        from mpiparticles.kernels.boris_velocity.boris_velocity_numpy import (
            boris_velocity as reference,
        )

        rng = np.random.default_rng(3)
        n = 40
        v0, e, b = (rng.normal(size=(3, n)) for _ in range(3))
        qm, dt = rng.normal(size=n), rng.uniform(0.05, 0.5, n)
        with fake_cupy_session():
            expected = v0.copy()
            reference(expected, e, b, qm, dt)
            v = cupy.array(v0, copy=True)
            boris_velocity(v, cupy.array(e), cupy.array(b), cupy.array(dt), cupy.array(qm))
            assert isinstance(v, cupy.ndarray)
            np.testing.assert_allclose(cupy.asnumpy(v), expected, rtol=0, atol=1e-13)

            expected = v0.copy()
            reference(expected, np.tile([[0.0], [0.0], [1.0]], n), b, np.full(n, 2.0),
                      np.full(n, 0.1))
            v = cupy.array(v0, copy=True)
            boris_velocity(v, [0.0, 0.0, 1.0], cupy.array(b), 0.1, 2.0)
            np.testing.assert_allclose(cupy.asnumpy(v), expected, rtol=0, atol=1e-13)

            for step, ratio in ((0.1, 2.0), (cupy.array(dt), cupy.array(qm))):
                store = ParticleStore(ndim=2, nvel=3)
                x0 = rng.normal(size=(2, n))
                store.add(cupy.array(x0, copy=True), cupy.array(v0, copy=True))
                boris_push(store, cupy.array(e), cupy.array(b), step, ratio)
                velocity = v0.copy()
                h = np.broadcast_to(cupy.asnumpy(cupy.asarray(step)), (n,)).copy()
                r = np.broadcast_to(cupy.asnumpy(cupy.asarray(ratio)), (n,)).copy()
                reference(velocity, e, b, r, h)
                got_x = cupy.asnumpy(store.positions[:, :n])
                got_v = cupy.asnumpy(store.velocities[:, :n])
                np.testing.assert_allclose(got_v, velocity, rtol=0, atol=1e-13)
                np.testing.assert_allclose(got_x, x0 + h * velocity[:2], rtol=0,
                                           atol=1e-13)
        """
    )
    run_in_fake_cupy_subprocess(code, timeout=600)


def _close(gpu, cpu, tol=3e-5):
    scale = max(float(np.abs(cpu).max()), 1e-30)
    np.testing.assert_allclose(gpu / scale, cpu / scale, rtol=0.0, atol=tol)


@METAL
def test_the_metal_velocity_kernel_agrees():
    args = _velocity_case(np.random.default_rng(4))
    expected, result = _copies(args), _copies(args)
    KERNELS["boris_velocity"](*expected)
    metal_kernel("boris_velocity")(*result)
    _close(result[0], expected[0])


@METAL
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_metal_push_kernel_agrees_and_keeps_float64_positions(ndim):
    args = _push_case(np.random.default_rng(5 + ndim), ndim)
    expected, result = _copies(args), _copies(args)
    KERNELS["push_boris"](*expected)
    metal_kernel("push_boris")(*result)
    _close(result[1], expected[1])
    # the displacement is float32, the positions (near 1000) stay float64
    np.testing.assert_allclose(result[0], expected[0], rtol=0.0, atol=1e-5)


def test_the_pyccel_boris_update_is_the_numpy_one():
    # the inlined helper of the Pyccel kernels (run in Python) against boris_update
    from mpiparticles.kernels.boris import boris_update as numpy_update
    from mpiparticles.kernels.boris_pyccel import boris_update

    rng = np.random.default_rng(6)
    v, kick, t = rng.normal(size=(3, 3, 40))
    expected = numpy_update(v, kick, t)
    got = np.array([boris_update(*v[:, k], *kick[:, k], *t[:, k]) for k in range(40)]).T
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-14)


def _pyccel_boris(v, kick, t):
    """The Pyccel block ``boris_update``, the source of truth, run in Python per marker."""
    from mpiparticles.kernels.boris_pyccel import boris_update

    columns = [boris_update(*v[:, i], *kick[:, i], *t[:, i]) for i in range(v.shape[1])]
    return np.array(columns, dtype=v.dtype).T


@EMULATOR
def test_boris_cuh_is_the_pyccel_block():
    # the hand-written CUDA version computes what the Pyccel block computes
    import cunumpy as xp

    source = """
#include "cunumpy/array_view.cuh"
#include "mpiparticles/kernels/boris.cuh"
extern "C" __global__ void run(Array2D<double> v, Array2D<double> k, Array2D<double> t,
                               Array2D<double> out) {
    const long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (i >= v.shape[1]) return;
    double a = v(0, i), b = v(1, i), c = v(2, i);
    mpiparticles_boris(a, b, c, k(0, i), k(1, i), k(2, i), t(0, i), t(1, i), t(2, i));
    out(0, i) = a;
    out(1, i) = b;
    out(2, i) = c;
}"""
    rng = np.random.default_rng(8)
    v, k, t = (rng.normal(size=(3, N)) for _ in range(3))
    roots = [str(p) for p in xp.kernels.source_roots("mpiparticles", "cunumpy")]
    out = np.zeros((3, N))
    emulate_cuda_kernel(
        xp.kernels.CudaKernel(source, "run", include_dirs=roots),
        v,
        k,
        t,
        out,
        n_threads=N,
    )
    np.testing.assert_allclose(out, _pyccel_boris(v, k, t), rtol=1e-14, atol=1e-14)


def test_the_numpy_boris_update_is_the_pyccel_block():
    from mpiparticles.kernels.boris import boris_update

    rng = np.random.default_rng(9)
    v, k, t = (rng.normal(size=(3, N)) for _ in range(3))
    np.testing.assert_allclose(
        boris_update(v, k, t), _pyccel_boris(v, k, t), rtol=1e-14, atol=1e-14
    )


@METAL
def test_the_metal_boris_header_is_the_pyccel_block():
    import cunumpy as xp

    from mpiparticles.kernels.boris import BORIS_METAL_HEADER

    source = """
    uint i = thread_position_in_grid.x;
    uint n = v_shape[1];
    if (i >= n) return;
    float3 r = mpiparticles_boris(float3(v[i], v[n + i], v[2 * n + i]),
                                  float3(k[i], k[n + i], k[2 * n + i]),
                                  float3(t[i], t[n + i], t[2 * n + i]));
    out[i] = r.x;
    out[n + i] = r.y;
    out[2 * n + i] = r.z;
"""
    kernel = xp.kernels.MetalKernel(
        source, ["v", "k", "t"], ["out"], name="boris_header", header=BORIS_METAL_HEADER
    )
    rng = np.random.default_rng(10)
    v, k, t = (rng.normal(size=(3, N)).astype(np.float32) for _ in range(3))
    out = np.zeros((3, N), np.float32)
    kernel(v, k, t, out=(out,), n_threads=N)
    _close(out, _pyccel_boris(v.astype(float), k.astype(float), t.astype(float)))


# The relativistic Boris kernels and blocks, in units where c ~ |u| so that gamma matters.
C = 1.3


def _relativistic_velocity_case(rng):
    return _velocity_case(rng) + [C]


def _relativistic_push_case(rng, ndim):
    return _push_case(rng, ndim) + [C]


@EMULATOR
def test_the_cuda_relativistic_velocity_kernel_agrees():
    args = _relativistic_velocity_case(np.random.default_rng(11))
    expected = _copies(args)
    KERNELS["relativistic_boris_velocity"].host_kernel.kernel.python(*expected)
    result = _copies(args)
    emulate_cuda_kernel(
        KERNELS["relativistic_boris_velocity"].cuda_kernel, *result, n_threads=N
    )
    np.testing.assert_allclose(result[0], expected[0], rtol=1e-13, atol=1e-13)


@EMULATOR
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_cuda_relativistic_push_kernel_agrees(ndim):
    args = _relativistic_push_case(np.random.default_rng(12 + ndim), ndim)
    expected = _copies(args)
    KERNELS["push_relativistic_boris"].host_kernel.kernel.python(*expected)
    result = _copies(args)
    emulate_cuda_kernel(
        KERNELS["push_relativistic_boris"].cuda_kernel, *result, n_threads=N
    )
    for i in (0, 1):
        np.testing.assert_allclose(result[i], expected[i], rtol=1e-13, atol=1e-13)


@EMULATOR
def test_the_relativistic_push_runs_on_cupy_arrays():
    code = textwrap.dedent(
        """
        import numpy as np
        from cunumpy.kernel_testing import fake_cupy_session

        import cupy  # the fake CuPy, installed by importing cunumpy
        from mpiparticles import ParticleStore, boris_push, boris_velocity
        from mpiparticles.kernels.relativistic_boris_velocity.relativistic_boris_velocity_numpy import (
            relativistic_boris_velocity as reference,
        )

        rng = np.random.default_rng(13)
        n, c = 40, 1.3
        u0, e, b = (rng.normal(size=(3, n)) for _ in range(3))
        qm, dt = rng.normal(size=n), rng.uniform(0.05, 0.5, n)
        with fake_cupy_session():
            expected = u0.copy()
            reference(expected, e, b, qm, dt, c)
            u = cupy.array(u0, copy=True)
            boris_velocity(u, cupy.array(e), cupy.array(b), cupy.array(dt), cupy.array(qm),
                           relativistic=True, speed_of_light=c)
            assert isinstance(u, cupy.ndarray)
            np.testing.assert_allclose(cupy.asnumpy(u), expected, rtol=0, atol=1e-13)

            for step, ratio in ((0.1, 2.0), (cupy.array(dt), cupy.array(qm))):
                store = ParticleStore(ndim=2, nvel=3)
                x0 = rng.normal(size=(2, n))
                store.add(cupy.array(x0, copy=True), cupy.array(u0, copy=True))
                boris_push(store, cupy.array(e), cupy.array(b), step, ratio,
                           relativistic=True, speed_of_light=c)
                velocity = u0.copy()
                h = np.broadcast_to(cupy.asnumpy(cupy.asarray(step)), (n,)).copy()
                r = np.broadcast_to(cupy.asnumpy(cupy.asarray(ratio)), (n,)).copy()
                reference(velocity, e, b, r, h, c)
                gamma = np.sqrt(1 + np.sum(velocity**2, axis=0) / c**2)
                moved = x0 + h / gamma * velocity[:2]
                got_x = cupy.asnumpy(store.positions[:, :n])
                got_u = cupy.asnumpy(store.velocities[:, :n])
                np.testing.assert_allclose(got_u, velocity, rtol=0, atol=1e-13)
                np.testing.assert_allclose(got_x, moved, rtol=0, atol=1e-13)
        """
    )
    run_in_fake_cupy_subprocess(code, timeout=600)


@METAL
def test_the_metal_relativistic_velocity_kernel_agrees():
    args = _relativistic_velocity_case(np.random.default_rng(14))
    expected, result = _copies(args), _copies(args)
    KERNELS["relativistic_boris_velocity"](*expected)
    metal_kernel("relativistic_boris_velocity")(*result)
    _close(result[0], expected[0])


@METAL
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_metal_relativistic_push_kernel_agrees(ndim):
    args = _relativistic_push_case(np.random.default_rng(15 + ndim), ndim)
    expected, result = _copies(args), _copies(args)
    KERNELS["push_relativistic_boris"](*expected)
    metal_kernel("push_relativistic_boris")(*result)
    _close(result[1], expected[1])
    np.testing.assert_allclose(result[0], expected[0], rtol=0.0, atol=1e-5)


def _pyccel_relativistic_boris(u, kick, b, inv_c2):
    """The Pyccel block ``relativistic_boris_update``, run in Python per marker."""
    from mpiparticles.kernels.boris_pyccel import relativistic_boris_update

    columns = [
        relativistic_boris_update(*u[:, i], *kick[:, i], *b[:, i], inv_c2)
        for i in range(u.shape[1])
    ]
    return np.array(columns, dtype=u.dtype).T


def test_the_numpy_relativistic_blocks_are_the_pyccel_blocks():
    from mpiparticles.kernels.boris import lorentz_factor, relativistic_boris_update
    from mpiparticles.kernels.boris_pyccel import lorentz_factor as pyccel_gamma

    rng = np.random.default_rng(16)
    u, k, b = (rng.normal(size=(3, N)) for _ in range(3))
    inv_c2 = 1 / C**2
    np.testing.assert_allclose(
        relativistic_boris_update(u, k, b, inv_c2),
        _pyccel_relativistic_boris(u, k, b, inv_c2),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        lorentz_factor(u, inv_c2),
        [pyccel_gamma(*u[:, i], inv_c2) for i in range(N)],
        rtol=1e-15,
    )


@EMULATOR
def test_the_relativistic_cuda_block_is_the_pyccel_block():
    import cunumpy as xp

    source = """
#include "cunumpy/array_view.cuh"
#include "mpiparticles/kernels/boris.cuh"
extern "C" __global__ void run(Array2D<double> u, Array2D<double> k, Array2D<double> b,
                               double inv_c2, Array2D<double> out) {
    const long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (i >= u.shape[1]) return;
    double x = u(0, i), y = u(1, i), z = u(2, i);
    mpiparticles_relativistic_boris(x, y, z, k(0, i), k(1, i), k(2, i), b(0, i), b(1, i),
                                    b(2, i), inv_c2);
    out(0, i) = x;
    out(1, i) = y;
    out(2, i) = z;
}"""
    rng = np.random.default_rng(17)
    u, k, b = (rng.normal(size=(3, N)) for _ in range(3))
    roots = [str(p) for p in xp.kernels.source_roots("mpiparticles", "cunumpy")]
    out = np.zeros((3, N))
    emulate_cuda_kernel(
        xp.kernels.CudaKernel(source, "run", include_dirs=roots),
        u,
        k,
        b,
        1 / C**2,
        out,
        n_threads=N,
    )
    np.testing.assert_allclose(
        out, _pyccel_relativistic_boris(u, k, b, 1 / C**2), rtol=1e-14, atol=1e-14
    )


@METAL
def test_the_metal_relativistic_header_is_the_pyccel_block():
    import cunumpy as xp

    from mpiparticles.kernels.boris import RELATIVISTIC_BORIS_METAL_HEADER

    source = """
    uint i = thread_position_in_grid.x;
    uint n = u_shape[1];
    if (i >= n) return;
    float3 r = mpiparticles_relativistic_boris(float3(u[i], u[n + i], u[2 * n + i]),
                                               float3(k[i], k[n + i], k[2 * n + i]),
                                               float3(b[i], b[n + i], b[2 * n + i]),
                                               inv_c2[0]);
    out[i] = r.x;
    out[n + i] = r.y;
    out[2 * n + i] = r.z;
"""
    kernel = xp.kernels.MetalKernel(
        source,
        ["u", "k", "b", "inv_c2"],
        ["out"],
        name="relativistic_boris_header",
        header=RELATIVISTIC_BORIS_METAL_HEADER,
    )
    rng = np.random.default_rng(18)
    u, k, b = (rng.normal(size=(3, N)).astype(np.float32) for _ in range(3))
    inv_c2 = np.array([1 / C**2], np.float32)
    out = np.zeros((3, N), np.float32)
    kernel(u, k, b, inv_c2, out=(out,), n_threads=N)
    _close(
        out,
        _pyccel_relativistic_boris(
            u.astype(float), k.astype(float), b.astype(float), float(inv_c2[0])
        ),
    )


@EMULATOR
def test_the_relativity_helpers_keep_the_array_kind():
    # with the CuPy backend a NumPy array stays a NumPy array (codes that trace on the host),
    # and a CuPy array stays on the device
    code = textwrap.dedent(
        """
        import numpy as np
        from cunumpy.kernel_testing import fake_cupy_session

        import cupy  # the fake CuPy, installed by importing cunumpy
        from mpiparticles import (
            coordinate_velocity, kinetic_energy, lorentz_factor, proper_velocity,
        )

        v = np.array([[0.6, 0.0], [0.0, 0.8], [0.0, 0.0]])
        with fake_cupy_session():
            for values, kind in ((v, np.ndarray), (cupy.array(v), cupy.ndarray)):
                u = proper_velocity(values, 1.0)
                for result in (
                    u, coordinate_velocity(u, 1.0), lorentz_factor(u, 1.0),
                    kinetic_energy(u, 2.0, 1.0),
                ):
                    assert type(result) is kind, (type(result), kind)
            np.testing.assert_allclose(lorentz_factor(proper_velocity(v, 1.0), 1.0),
                                       [1.25, 5 / 3])
        """
    )
    run_in_fake_cupy_subprocess(code, timeout=600)
