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
