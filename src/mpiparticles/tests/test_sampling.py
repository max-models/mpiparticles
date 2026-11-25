"""Tests for the velocity draws and the importance sampling."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest
from scipy import special

from mpiparticles.sampling import (
    ImportanceSampling,
    PositionBias,
    as_generator,
    flux_speeds,
    make_generator,
    maxwellian_velocities,
    normal,
    uniform,
)


def _host(array) -> np.ndarray:
    return np.asarray(xp.to_numpy(array) if xp.is_gpu(array) else array)


@pytest.fixture
def rng():
    return make_generator(1234)


def test_generators_from_seeds_are_reproducible() -> None:
    a = as_generator(5).random(4)
    b = as_generator(make_generator(5)).random(4)
    np.testing.assert_array_equal(_host(a), _host(b))
    assert as_generator(None).random(3).shape == (3,)
    host = make_generator(1, backend="numpy")
    assert isinstance(host, np.random.Generator)
    assert normal(host, 1.0, 0.0, 3).tolist() == [1.0, 1.0, 1.0]
    assert np.all(uniform(host, 2.0, 3.0, 100) >= 2.0)


def test_maxwellian_and_flux_draws_have_their_moments(rng) -> None:
    velocities = _host(maxwellian_velocities(rng, 200_000, 3, 2.0))
    np.testing.assert_allclose(velocities.std(axis=1), 2.0, rtol=0.01)
    speeds = _host(flux_speeds(rng, 200_000, 2.0))
    assert np.all(speeds >= 0.0)
    assert speeds.mean() == pytest.approx(2.0 * np.sqrt(np.pi / 2.0), rel=0.01)


def test_trivial_sampling_has_unit_factors(rng) -> None:
    sampling = ImportanceSampling()
    assert sampling.is_trivial
    _, factors = sampling.sample_maxwellian(rng, 100, 3, 1.0)
    np.testing.assert_allclose(_host(factors), 1.0)
    speeds, factors = sampling.sample_flux_speeds(rng, 10, 1.0)
    np.testing.assert_allclose(_host(factors), 1.0)


@pytest.mark.parametrize("nvel", [1, 3])
def test_hot_maxwellian_moments_are_unbiased(rng, nvel: int) -> None:
    v_th = 2.0
    sampling = ImportanceSampling(temperature_ratio=4.0)
    velocities, factors = sampling.sample_maxwellian(rng, 200_000, nvel, v_th)
    velocities, factors = _host(velocities), _host(factors)
    # samples come from the hot distribution...
    assert np.std(velocities[0]) == pytest.approx(2.0 * v_th, rel=0.02)
    # ...but the weighted moments are those of f
    assert np.mean(factors) == pytest.approx(1.0, rel=0.01)
    weighted_var = np.sum(factors * velocities**2, axis=1) / factors.size
    np.testing.assert_allclose(weighted_var, v_th**2, rtol=0.03)
    # the weights are bounded by temperature_ratio ** (nvel / 2)
    assert factors.max() <= 4.0 ** (nvel / 2) + 1e-12


def test_drifted_sampling_recovers_zero_mean_velocity(rng) -> None:
    sampling = ImportanceSampling(temperature_ratio=2.0, drift=(1.5,))
    assert sampling.has_drift and sampling.biases_velocities
    velocities, factors = sampling.sample_maxwellian(rng, 200_000, 1, 1.0)
    velocities, factors = _host(velocities)[0], _host(factors)
    assert np.mean(velocities) == pytest.approx(1.5, abs=0.02)
    assert np.mean(factors * velocities) == pytest.approx(0.0, abs=0.02)
    assert np.mean(factors) == pytest.approx(1.0, rel=0.02)


def test_drift_length_must_match_the_velocity_components(rng) -> None:
    sampling = ImportanceSampling(drift=(1.0, 0.0))
    with pytest.raises(ValueError, match="drift has 2 components"):
        sampling.sample_maxwellian(rng, 10, 3, 1.0)


def test_tail_estimate_has_lower_variance_with_hot_sampling() -> None:
    """The point of the feature: less noisy estimates of a rare tail."""
    v_th, cut, num = 1.0, 3.0, 2000
    exact = special.erfc(cut / np.sqrt(2.0))  # P(|v| > 3 v_th) in 1-D

    def estimates(sampling: ImportanceSampling) -> np.ndarray:
        values = []
        for seed in range(40):
            v, w = sampling.sample_maxwellian(make_generator(seed), num, 1, v_th)
            values.append(np.sum(_host(w)[np.abs(_host(v)[0]) > cut]) / num)
        return np.asarray(values)

    plain = estimates(ImportanceSampling())
    hot = estimates(ImportanceSampling(temperature_ratio=4.0))
    assert np.mean(hot) == pytest.approx(exact, rel=0.1)
    assert np.std(hot) < 0.5 * np.std(plain)


def test_flux_speeds_are_unbiased(rng) -> None:
    sampling = ImportanceSampling(temperature_ratio=3.0)
    speeds, factors = sampling.sample_flux_speeds(rng, 200_000, 1.0)
    speeds, factors = _host(speeds), _host(factors)
    assert np.all(speeds >= 0.0)
    assert np.mean(factors) == pytest.approx(1.0, rel=0.01)
    # the mean of the flux-weighted half-Maxwellian (Rayleigh) is v_th sqrt(pi/2)
    assert np.mean(factors * speeds) == pytest.approx(np.sqrt(np.pi / 2.0), rel=0.01)
    transverse, factors = sampling.sample_transverse(rng, 1000, 1.0)
    assert transverse.shape == factors.shape == (1000,)


def test_flux_sampling_rejects_a_drift(rng) -> None:
    with pytest.raises(ValueError, match="drift is not available"):
        ImportanceSampling(drift=(1.0,)).sample_flux_speeds(rng, 10, 1.0)


def test_temperature_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError, match="temperature_ratio"):
        ImportanceSampling(temperature_ratio=0.0)


def test_position_bias_mean_is_exact() -> None:
    bias = PositionBias(points=[0.0, 0.5, 1.0], values=[0.1, 0.1, 1.0])
    assert bias.axis == 0
    np.testing.assert_array_equal(bias.values, [0.1, 0.1, 1.0])
    # 0.5 * 0.1 + 0.5 * (0.1 + 1.0) / 2 = 0.325
    assert bias.mean_over_box((0.0,), (1.0,)) == pytest.approx(0.325)
    # constant outside the points
    assert bias.mean_over_box((-1.0,), (0.0,)) == pytest.approx(0.1)
    assert bias.mean_over_box((0.75,), (0.75,)) == pytest.approx(0.55)
    assert ImportanceSampling(position_bias=bias).mean_bias(
        lambda m: None, (0.0,), (1.0,)
    ) == pytest.approx(0.325)


@pytest.mark.parametrize(
    ("points", "values", "axis"),
    [
        ([0.0, 1.0], [0.0, 1.0], 0),
        ([0.0, 1.0], [0.5, 1.5], 0),
        ([1.0, 0.0], [1.0, 1.0], 0),
        ([0.0], [1.0, 1.0], 0),
        ([0.0, 1.0], [1.0, 1.0], -1),
    ],
)
def test_position_bias_validates_its_table(points, values, axis) -> None:
    with pytest.raises(ValueError):
        PositionBias(points=points, values=values, axis=axis)


def test_callable_bias_out_of_range_is_rejected(rng) -> None:
    sampling = ImportanceSampling(position_bias=lambda x: 2.0 * xp.ones(x.shape[1]))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        sampling.sample_positions(rng, 10, lambda m: rng.random((1, m)), mean_bias=1.0)


def test_biased_positions_follow_g_but_weighted_density_is_uniform(rng) -> None:
    bias = PositionBias(points=[0.0, 1.0], values=[0.1, 1.0])
    sampling = ImportanceSampling(position_bias=bias)
    assert sampling.biases_positions and not sampling.is_trivial
    positions, factors = sampling.sample_positions(
        rng,
        100_000,
        lambda m: rng.random((1, m)),
        mean_bias=bias.mean_over_box((0.0,), (1.0,)),
    )
    x, w = _host(positions)[0], _host(factors)
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(x, edges)
    weighted, _ = np.histogram(x, edges, weights=w)
    # the marker count follows g...
    assert counts[-1] / counts[0] == pytest.approx(0.955 / 0.145, rel=0.1)
    # ...while the weighted density is flat, with the expected total
    np.testing.assert_allclose(weighted / (x.size / 10), 1.0, rtol=0.06)
    assert np.mean(w) == pytest.approx(1.0, rel=0.01)


def test_mean_bias_estimate_for_a_callable(rng) -> None:
    sampling = ImportanceSampling(position_bias=lambda x: 0.25 + 0.5 * x[0])
    estimate = sampling.mean_bias(lambda m: rng.random((1, m)))
    assert estimate == pytest.approx(0.5, rel=0.01)
    assert ImportanceSampling().mean_bias(lambda m: None) == 1.0
    positions, factors = ImportanceSampling().sample_positions(
        rng, 5, lambda m: rng.random((1, m)), 1.0
    )
    assert positions.shape == (1, 5)
    np.testing.assert_allclose(_host(factors), 1.0)
