"""Velocity moments and phase-space densities (serially; the MPI job sums over ranks)."""

from __future__ import annotations

from math import comb

import cunumpy as xp
import numpy as np
import pytest

from mpiparticles import (
    BinAxis,
    moment_indices,
    moment_name,
    moment_values,
    phase_space_density,
    velocity_moments,
)
from mpiparticles.moments import bin_axes
from mpiparticles.tests.conftest import FakeComm, filled_store


def _markers(count: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 1.0, (2, count))
    velocities = rng.normal(0.3, 1.0, (3, count))
    weights = rng.uniform(0.5, 1.5, count)
    return positions, velocities, weights


def test_moment_indices_lists_each_symmetric_moment_once() -> None:
    assert moment_indices(3, 2) == (
        (),
        (0,),
        (1,),
        (2,),
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    )
    for nvel in (1, 2, 3):
        for order in range(5):
            count = sum(comb(nvel + k - 1, k) for k in range(order + 1))
            assert len(moment_indices(nvel, order)) == count
    assert [moment_name(i) for i in moment_indices(1, 3)] == ["", "x", "xx", "xxx"]


@pytest.mark.parametrize(("nvel", "order"), [(0, 1), (3, -1)])
def test_moment_indices_rejects_bad_arguments(nvel: int, order: int) -> None:
    with pytest.raises(ValueError):
        moment_indices(nvel, order)


def test_moment_values_are_the_products_of_the_velocities() -> None:
    _, velocities, weights = _markers(50)
    values = np.asarray(xp.to_numpy(moment_values(velocities, weights, 3)))
    for row, index in zip(values, moment_indices(3, 3), strict=True):
        expected = weights * np.prod([velocities[i] for i in index], axis=0)
        np.testing.assert_allclose(row, expected, rtol=1e-14)
    assert moment_values(velocities[0], weights, 2).shape == (3, 50)
    with pytest.raises(ValueError, match="markers"):
        moment_values(velocities, weights[:-1])


def test_velocity_moments_of_a_store() -> None:
    store = filled_store(5)  # velocity -i - d, weight i + 1
    sums = velocity_moments(store.active_velocities(), store.active_weights(), 2)
    weights = np.arange(5) + 1.0
    vx, vy = -np.arange(5.0), -np.arange(5.0) - 1
    names = [moment_name(i) for i in moment_indices(3, 2)]
    values = dict(zip(names, sums, strict=True))
    assert values[""] == pytest.approx(weights.sum())
    assert values["x"] == pytest.approx((weights * vx).sum())
    assert values["xy"] == pytest.approx((weights * vx * vy).sum())
    assert velocity_moments(np.zeros((3, 0)), np.zeros(0), 1).tolist() == [0.0] * 4


def test_serial_reductions_make_no_mpi_call() -> None:
    class NoCommunication(FakeComm):
        def __getattr__(self, name):
            raise AssertionError(f"MPI call {name} in a serial run")

    positions, velocities, weights = _markers(10)
    comm = NoCommunication(0, 1)
    velocity_moments(velocities, weights, 2, comm)
    phase_space_density(positions, velocities, weights, {"vx": (-3, 3, 4)}, comm)


def test_the_phase_space_density_integrates_to_the_weight_inside() -> None:
    positions, velocities, weights = _markers()
    axes = {"x": (0.0, 1.0, 10), "vx": (-1.0, 2.0, 12)}
    density = phase_space_density(positions, velocities, weights, axes)
    assert isinstance(density, np.ndarray)
    assert density.shape == (10, 12)
    inside = (velocities[0] >= -1.0) & (velocities[0] <= 2.0)
    total = density.sum() * 0.1 * 0.25
    assert total == pytest.approx(weights[inside].sum(), rel=1e-12)


def test_the_phase_space_density_matches_numpy() -> None:
    positions, velocities, weights = _markers()
    axes = (BinAxis("y", 0.0, 1.0, 5), BinAxis("vz", -2.0, 2.0, 8))
    density = phase_space_density(positions, velocities, weights, axes)
    expected, _, _ = np.histogram2d(
        positions[1],
        velocities[2],
        bins=[axes[0].edges, axes[1].edges],
        weights=weights,
    )
    np.testing.assert_allclose(density, expected / (0.2 * 0.5), rtol=1e-12)


def test_bin_axes() -> None:
    axis = BinAxis("vx", -1.0, 1.0, 4)
    assert axis.is_velocity and axis.component == 0 and axis.units == "m/s"
    assert BinAxis("z", 0.0, 1.0, 2).units == "m"
    np.testing.assert_allclose(axis.centers, [-0.75, -0.25, 0.25, 0.75])
    assert axis.width == 0.5
    assert bin_axes({"x": (0, 1, 3)}) == (BinAxis("x", 0.0, 1.0, 3),)
    with pytest.raises(ValueError, match="one to three"):
        bin_axes({})
    with pytest.raises(ValueError, match="unknown"):
        BinAxis("w", 0.0, 1.0, 2)
    with pytest.raises(ValueError, match="upper > lower"):
        BinAxis("x", 1.0, 1.0, 2)
    with pytest.raises(ValueError, match="num_bins"):
        BinAxis("x", 0.0, 1.0, 0)


def test_an_axis_the_markers_lack_raises() -> None:
    positions, velocities, weights = _markers(10)
    with pytest.raises(ValueError, match="position component 2"):
        phase_space_density(positions, velocities, weights, {"z": (0, 1, 2)})
