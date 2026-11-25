"""Tests for the timestep sources of `mpiparticles.injection`."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest
from scipy import constants
from simgeometry.boundingbox import BoundingBox
from simgeometry.objects.aperture2d import Aperture2D
from simgeometry.objects.circle import Circle
from simgeometry.objects.polygon import Polygon

from mpiparticles import ParticleStore
from mpiparticles.injection import (
    ChargeBalanceSource,
    DirectedSource,
    MarkerSpecies,
    PlanarFluxInjector1D,
    PlanarFluxInjector2D,
    ProfiledVolumeSource,
    ReemissionSource,
    SurfaceConversionSource,
    SurfaceSource,
    UniformVolumeSource,
    cosine_profile,
    object_surface_segments,
    planar_flux_injector,
)
from mpiparticles.sampling import ImportanceSampling, PositionBias

ELECTRON_MASS = constants.m_e
PROTON_MASS = constants.m_p


def _host(array) -> np.ndarray:
    return np.asarray(xp.to_numpy(array) if xp.is_gpu(array) else array)


class Species:
    """A store with its species, as a code keeps them."""

    def __init__(
        self,
        ndim: int,
        nvel: int,
        weight: float = 1.0,
        temperature_ev: float = 1.0,
        mass: float = ELECTRON_MASS,
        charge: float = -constants.e,
    ) -> None:
        self.info = MarkerSpecies(
            charge=charge,
            mass=mass,
            weight=weight,
            thermal_velocity=float(np.sqrt(constants.e * temperature_ev / mass)),
            ndim=ndim,
            nvel=nvel,
        )
        self.store = ParticleStore(ndim=ndim, nvel=nvel)

    def inject(self, source, dt: float) -> int:
        return source.inject(self.store, dt, self.info)

    @property
    def v_th(self) -> float:
        return self.info.thermal_velocity

    @property
    def positions(self) -> np.ndarray:
        return _host(self.store.positions[:, : self.store.length])

    @property
    def velocities(self) -> np.ndarray:
        return _host(self.store.velocities[:, : self.store.length])

    @property
    def weights(self) -> np.ndarray:
        return _host(self.store.weights[: self.store.length])


def test_marker_species_validates_and_converts_temperatures() -> None:
    species = MarkerSpecies(charge=1.0, mass=2.0, weight=1.0, thermal_velocity=3.0)
    assert species.thermal_velocity_at(None) == 3.0
    assert species.thermal_velocity_at(2.0) == pytest.approx(np.sqrt(constants.e))
    for bad in ({"weight": 0.0}, {"mass": 0.0}, {"thermal_velocity": -1.0}):
        with pytest.raises(ValueError):
            MarkerSpecies(**{"charge": 1.0, "mass": 1.0, "weight": 1.0, **bad})


def test_inject_checks_the_store_components() -> None:
    species = Species(ndim=1, nvel=1)
    source = PlanarFluxInjector1D(position=0.0, current_density=1.0)
    with pytest.raises(ValueError, match="components"):
        source.inject(ParticleStore(ndim=1, nvel=3), 1.0, species.info)


# Planar flux


def test_planar_flux_1d_tracks_the_current_density() -> None:
    species = Species(ndim=1, nvel=1, weight=1.0e3)
    current_density = 2.0
    source = PlanarFluxInjector1D(position=0.25, current_density=current_density, rng=1)
    dt = 1.0e-15
    created = sum(species.inject(source, dt) for _ in range(500))
    expected = current_density * dt * 500 / constants.e / 1.0e3
    assert abs(created - expected) <= 1.0 + 1e-9
    np.testing.assert_allclose(species.positions, 0.25)
    # balanced pairs: as many go left as right
    assert abs(np.sum(np.sign(species.velocities[0]))) <= 1
    assert source.position == 0.25 and source.direction is None
    assert source.current_density == current_density and source.sampling is None
    assert source.bounding_box.lower_bound == (0.25,)


def test_planar_flux_1d_in_a_fixed_direction_has_the_flux_speeds() -> None:
    species = Species(ndim=1, nvel=3, weight=1.0)
    source = PlanarFluxInjector1D(
        position=0.0,
        current_density=1.0e5 * constants.e,
        rng=2,
        direction=-1.0,
    )
    assert species.inject(source, 1.0) == 100_000
    v = species.velocities
    assert np.all(v[0] < 0.0)
    np.testing.assert_array_equal(v[1:], 0.0)  # padded components
    assert np.mean(-v[0]) == pytest.approx(species.v_th * np.sqrt(np.pi / 2), rel=0.01)


def test_planar_flux_1d_checks() -> None:
    with pytest.raises(ValueError):
        PlanarFluxInjector1D(position=0.0, current_density=-1.0)
    with pytest.raises(ValueError):
        PlanarFluxInjector1D(position=0.0, current_density=1.0, direction=0.0)
    source = PlanarFluxInjector1D(position=0.0, current_density=1.0)
    for args in ((1.0, 1.0, -1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)):
        with pytest.raises(ValueError):
            source.markers_per_step(*args)
    with pytest.raises(ValueError, match="1-D"):
        source.emit(1.0, Species(ndim=2, nvel=2).info)


def test_planar_flux_thermalized_velocities_are_balanced_left_right() -> None:
    source = PlanarFluxInjector1D(position=0.0, current_density=1.0, rng=3)
    first = _host(source.sample_thermalized_velocities(3, 1.0))
    second = _host(source.sample_thermalized_velocities(3, 1.0))
    signs = np.sign(np.concatenate([first[0], second[0]]))
    assert signs.tolist() == [1, -1, 1, -1, 1, -1]
    assert source.sample_thermalized_velocities(0, 1.0).shape == (1, 0)
    with pytest.raises(ValueError):
        source.sample_thermalized_velocities(-1, 1.0)
    with pytest.raises(ValueError):
        source.sample_thermalized_velocities(1, -1.0)
    directed = PlanarFluxInjector1D(position=0.0, current_density=1.0, direction=1.0)
    assert np.all(_host(directed.sample_thermalized_velocities(10, 1.0)) > 0.0)


def test_planar_flux_1d_with_sampling_preserves_the_current() -> None:
    species = Species(ndim=1, nvel=1)
    source = PlanarFluxInjector1D(
        position=0.0,
        current_density=1.0e5 * constants.e,
        direction=1.0,
        rng=7,
        sampling=ImportanceSampling(temperature_ratio=4.0),
    )
    assert species.inject(source, 1.0) == 100_000
    v, w = species.velocities[0], species.weights
    assert np.all(v > 0.0)
    assert np.sum(w) == pytest.approx(1.0e5, rel=0.02)
    assert np.sum(w * v) / np.sum(w) == pytest.approx(
        species.v_th * np.sqrt(np.pi / 2.0), rel=0.02
    )
    # the hotter sampling puts many more markers above 3 v_th
    assert np.mean(v > 3.0 * species.v_th) > 5 * np.exp(-4.5)


def test_planar_flux_2d_emits_on_the_line() -> None:
    species = Species(ndim=2, nvel=3, weight=1.0)
    source = PlanarFluxInjector2D(
        x_source=0.1,
        y_lo=0.2,
        y_hi=0.7,
        current_density=2.0e5 * constants.e,
        inject_direction=None,
        rng=4,
    )
    assert species.inject(source, 1.0) in (99_999, 100_000)
    x, y = species.positions
    np.testing.assert_allclose(x, 0.1)
    assert y.min() >= 0.2 and y.max() <= 0.7
    v = species.velocities
    assert abs(np.sum(np.sign(v[0]))) <= 1
    assert np.std(v[1]) == pytest.approx(species.v_th, rel=0.02)
    np.testing.assert_array_equal(v[2], 0.0)
    assert source.position == 0.1 and source.direction is None
    assert source.current_density == pytest.approx(2.0e5 * constants.e)
    assert source.sampling is None and source.bounding_box.ndim == 2
    assert species.inject(source, 0.0) == 0


def test_planar_flux_2d_cold_and_cylindrical() -> None:
    cold = PlanarFluxInjector2D(0.0, 0.0, 1.0, constants.e * 1.0e3, transverse="cold")
    _, velocities = cold.emit(1.0, Species(ndim=2, nvel=2).info)[:2]
    np.testing.assert_array_equal(_host(velocities[1]), 0.0)
    assert _host(cold.sample_thermalized_velocities(4, 1.0)).shape == (2, 4)
    ring = PlanarFluxInjector2D(
        0.0, 0.0, 1.0, constants.e * 1.0e4, rng=5, rethermalize=True
    )
    ring.cylindrical = True
    assert ring.rethermalizes
    species = Species(ndim=2, nvel=3)
    created = species.inject(ring, 1.0)
    assert created == int(np.floor(np.pi * 1.0e4))
    # rings weighted by r: a uniform current density over the disc
    assert np.sum(species.weights) == pytest.approx(created, rel=0.02)
    assert np.std(species.velocities[2]) == pytest.approx(species.v_th, rel=0.05)
    assert _host(ring.sample_thermalized_velocities(4, 1.0)).shape == (3, 4)
    ring.flux_fraction = 0.5
    sampled = PlanarFluxInjector2D(
        0.0, 0.0, 1.0, constants.e * 1.0e4, sampling=ImportanceSampling()
    )
    sampled.cylindrical = True
    with pytest.raises(NotImplementedError):
        sampled.emit(1.0, species.info)


def test_planar_flux_2d_checks() -> None:
    for args, kwargs in (
        ((0.0, 0.0, 1.0, -1.0), {}),
        ((0.0, 1.0, 0.0, 1.0), {}),
        ((0.0, 0.0, 1.0, 1.0, 0.0), {}),
        ((0.0, 0.0, 1.0, 1.0), {"transverse": "warm"}),
    ):
        with pytest.raises(ValueError):
            PlanarFluxInjector2D(*args, **kwargs)
    source = PlanarFluxInjector2D(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="2-D"):
        source.emit(1.0, Species(ndim=1, nvel=1).info)
    for args in ((-1, 1.0), (1, -1.0)):
        with pytest.raises(ValueError):
            source.sample_thermalized_velocities(*args)


@pytest.mark.parametrize("transverse", ["thermal", "cold"])
def test_planar_flux_2d_with_sampling_biases_y(transverse: str) -> None:
    species = Species(ndim=2, nvel=2)
    bias = PositionBias(points=[0.0, 1.0], values=[0.1, 1.0], axis=1)
    source = PlanarFluxInjector2D(
        x_source=0.0,
        y_lo=0.0,
        y_hi=1.0,
        current_density=1.0e5 * constants.e,
        rng=11,
        transverse=transverse,
        sampling=ImportanceSampling(temperature_ratio=2.0, position_bias=bias),
    )
    assert species.inject(source, 1.0) == 100_000
    positions, velocities, w = species.positions, species.velocities, species.weights
    np.testing.assert_allclose(positions[0], 0.0)
    assert np.sum(w) == pytest.approx(1.0e5, rel=0.03)
    if transverse == "thermal":
        assert np.sum(w * velocities[1] ** 2) / np.sum(w) == pytest.approx(
            species.v_th**2, rel=0.04
        )
    upper, lower = positions[1] > 0.9, positions[1] < 0.1
    assert np.count_nonzero(upper) > 4 * np.count_nonzero(lower)
    # weighted, both ends carry the same flux
    assert np.sum(w[upper]) == pytest.approx(np.sum(w[lower]), rel=0.15)


def test_planar_flux_injector_on_a_box() -> None:
    one = planar_flux_injector(BoundingBox(lower_bound=(0.5,), upper_bound=(1.0,)), 1.0)
    assert isinstance(one, PlanarFluxInjector1D) and one.position == 0.5
    two = planar_flux_injector(
        BoundingBox(lower_bound=(0.5, 0.0), upper_bound=(1.0, 2.0)), 1.0
    )
    assert isinstance(two, PlanarFluxInjector2D)
    assert two.bounding_box.upper_bound == (0.5, 2.0)
    with pytest.raises(NotImplementedError):
        planar_flux_injector(
            BoundingBox(lower_bound=(0.0,) * 3, upper_bound=(1.0,) * 3), 1.0
        )


# Volume sources


def test_volume_rate_is_exact_on_average_and_markers_lie_in_the_box() -> None:
    species = Species(ndim=2, nvel=2, weight=1.0e6, mass=PROTON_MASS)
    box = BoundingBox(lower_bound=(0.1, 0.2), upper_bound=(0.4, 0.9))
    rate = 3.7e15
    source = UniformVolumeSource(bounding_box=box, rate=rate, rng=1)
    dt = 1.0e-10
    created = sum(species.inject(source, dt) for _ in range(500))
    assert abs(created - rate * dt * 500 / 1.0e6) <= 1.0 + 1e-9
    positions = species.positions
    assert positions.shape[1] == created
    assert np.all((positions[0] >= 0.1) & (positions[0] <= 0.4))
    assert np.all((positions[1] >= 0.2) & (positions[1] <= 0.9))
    assert source.bounding_box is box and source.sampling is None


def test_volume_velocities_are_maxwellian_and_the_rate_can_change() -> None:
    species = Species(ndim=2, nvel=2)
    box = BoundingBox(lower_bound=(0.0, 0.0), upper_bound=(1.0, 1.0))
    source = UniformVolumeSource(bounding_box=box, rate=2.0e5, rng=3)
    species.inject(source, 1.0)
    velocities = species.velocities
    assert velocities.shape == (2, 200_000)
    np.testing.assert_allclose(velocities.std(axis=1), species.v_th, rtol=0.02)
    source.rate = 10.0
    assert source.rate == 10.0
    with pytest.raises(ValueError):
        source.rate = float("nan")
    source.flux_fraction = 0.5
    assert source.flux_fraction == 0.5
    assert species.inject(source, 1.0) == 5
    with pytest.raises(ValueError):
        source.flux_fraction = 0.0
    with pytest.raises(ValueError):
        UniformVolumeSource(bounding_box=box, rate=-1.0)
    with pytest.raises(ValueError, match="box is 2-D"):
        source.emit(1.0, Species(ndim=1, nvel=1).info)


def test_volume_fractional_markers_carry_over() -> None:
    species = Species(ndim=2, nvel=2)
    box = BoundingBox(lower_bound=(0.0, 0.0), upper_bound=(1.0, 1.0))
    source = UniformVolumeSource(bounding_box=box, rate=0.3)
    counts = [species.inject(source, 1.0) for _ in range(10)]
    assert sum(counts) in (2, 3)
    assert max(counts) == 1


def test_cylindrical_volume_rings_are_weighted_by_radius() -> None:
    species = Species(ndim=2, nvel=3)
    box = BoundingBox(lower_bound=(0.0, 0.0), upper_bound=(1.0, 1.0))
    source = UniformVolumeSource(bounding_box=box, rate=1.0e5, rng=2)
    source.cylindrical = True
    species.inject(source, 1.0)
    np.testing.assert_allclose(species.weights, 2.0 * species.positions[1])
    sampled = UniformVolumeSource(box, rate=10.0, sampling=ImportanceSampling())
    sampled.cylindrical = True
    with pytest.raises(NotImplementedError):
        sampled.emit(1.0, species.info)


def test_volume_source_with_sampling_preserves_the_rate() -> None:
    species = Species(ndim=2, nvel=2)
    box = BoundingBox(lower_bound=(0.0, 0.0), upper_bound=(1.0, 1.0))
    bias = PositionBias(points=[0.0, 1.0], values=[1.0, 0.25], axis=1)
    sampling = ImportanceSampling(temperature_ratio=2.0, position_bias=bias)
    source = UniformVolumeSource(bounding_box=box, rate=1.0e5, rng=3, sampling=sampling)
    assert source.sampling is sampling
    assert species.inject(source, 1.0) == 100_000
    positions, w = species.positions, species.weights
    assert np.sum(w) == pytest.approx(1.0e5, rel=0.02)
    assert np.count_nonzero(positions[1] < 0.1) > 2 * np.count_nonzero(
        positions[1] > 0.9
    )


def _cosine_source(**kwargs) -> ProfiledVolumeSource:
    box = BoundingBox(lower_bound=(0.2, 0.0), upper_bound=(0.6, 1.0))
    points, values = cosine_profile(0.2, 0.6)
    return ProfiledVolumeSource(
        box, rate=kwargs.pop("rate", 4.0e5), points=points, values=values, **kwargs
    )


def test_profiled_markers_follow_the_cosine_and_the_rate_is_the_total() -> None:
    species = Species(ndim=2, nvel=3, temperature_ev=10.0)
    assert species.inject(_cosine_source(rng=1), 1.0) == 400_000
    x, y = species.positions
    assert x.min() >= 0.2 and x.max() <= 0.6
    # on s in [-pi/2, pi/2] with density cos(s): <s^2> = pi^2/4 - 2
    s = (x - 0.4) / 0.4 * np.pi
    assert np.mean(s**2) == pytest.approx(np.pi**2 / 4 - 2.0, rel=0.01)
    assert np.mean(y) == pytest.approx(0.5, abs=0.005)


def test_sources_with_alike_pair_generators_create_pairs_at_the_same_positions() -> (
    None
):
    electrons = Species(ndim=2, nvel=3, temperature_ev=10.0)
    ions = Species(ndim=2, nvel=3, temperature_ev=10.0, mass=131.0 * PROTON_MASS)
    for species, seed in ((electrons, 1), (ions, 2)):
        species.inject(_cosine_source(rng=seed, pair_rng=9, rate=1000.0), 1.0)
    assert np.array_equal(electrons.positions, ions.positions)
    assert np.std(electrons.velocities) / np.std(ions.velocities) == pytest.approx(
        np.sqrt(131.0 * PROTON_MASS / ELECTRON_MASS), rel=0.15
    )


@pytest.mark.parametrize(
    ("points", "values"),
    [
        ([0.0, 1.0], [1.0, -1.0]),
        ([1.0, 0.0], [1.0, 1.0]),
        ([0.0, 1.0], [0.0, 0.0]),
        ([0.0], [1.0]),
    ],
)
def test_invalid_profiles_are_rejected(points, values) -> None:
    box = BoundingBox(lower_bound=(0.0, 0.0), upper_bound=(1.0, 1.0))
    with pytest.raises(ValueError):
        ProfiledVolumeSource(box, 1.0, points=points, values=values)


# Wall-driven sources


def test_the_emission_plane_carries_away_the_absorbed_charge() -> None:
    electrons = Species(ndim=2, nvel=3, weight=10.0)
    absorbed = {"q": 0.0}
    source = ChargeBalanceSource(
        position=0.8,
        y_range=(0.0, 1.0),
        absorbed_charge=lambda: absorbed["q"],
        temperature_ev=10.0,
        rng=3,
    )
    charge = electrons.info.charge
    assert electrons.inject(source, 1e-9) == 0
    absorbed["q"] = 1000 * charge
    assert electrons.inject(source, 1e-9) == 100
    assert source.emitted_weight == pytest.approx(1000.0)
    assert electrons.inject(source, 1e-9) == 0
    # net positive charge at the anode: the deficit is made up before the next emission
    absorbed["q"] = 500 * charge
    assert electrons.inject(source, 1e-9) == 0
    absorbed["q"] = 1500 * charge
    assert electrons.inject(source, 1e-9) == 50
    assert np.all(electrons.positions[0] == 0.8) and source.position == 0.8


def test_charge_balance_emits_a_full_maxwellian_at_its_temperature() -> None:
    electrons = Species(ndim=2, nvel=3)
    source = ChargeBalanceSource(
        0.5, (0.0, 1.0), lambda: 100_000 * electrons.info.charge, temperature_ev=25.0
    )
    electrons.inject(source, 1e-9)
    v = electrons.velocities
    expected = np.sqrt(25.0 * constants.e / ELECTRON_MASS)
    assert v.shape == (3, 100_000)
    assert np.std(v[0]) == pytest.approx(expected, rel=0.02)
    assert abs(np.mean(v[0])) < 0.02 * expected
    with pytest.raises(ValueError):
        ChargeBalanceSource(0.5, (1.0, 0.0), lambda: 0.0)
    with pytest.raises(ValueError):
        source.emit(1.0, Species(ndim=1, nvel=1).info)


def test_reemission_net_current_alone_emits_at_the_set_rate() -> None:
    electrons = Species(ndim=1, nvel=3, weight=1.0e6)
    source = ReemissionSource(position=1e-6, net_current_density=0.03, rng=1)
    assert source.position == 1e-6 and source.net_current_density == 0.03
    emitted = sum(electrons.inject(source, 1.0e-9) for _ in range(1000))
    assert emitted == pytest.approx(0.03 / constants.e * 1.0e-6 / 1.0e6, abs=1)
    assert source.emitted_weight == pytest.approx(emitted * 1.0e6)


def test_every_absorbed_particle_is_emitted_again() -> None:
    electrons = Species(ndim=1, nvel=3, weight=10.0)
    absorbed = {"w": 0.0}
    source = ReemissionSource(position=0.0, absorbed_weight=lambda: absorbed["w"])
    assert electrons.inject(source, 1e-9) == 0
    absorbed["w"] = 1000.0
    assert electrons.inject(source, 1e-9) == 100
    absorbed["w"] = 1500.0
    assert electrons.inject(source, 1e-9) == 50


def test_reemission_velocities_are_a_flux_pointing_away_from_the_wall() -> None:
    electrons = Species(ndim=1, nvel=3)
    source = ReemissionSource(
        position=0.0,
        absorbed_weight=lambda: 200_000.0,
        direction=1.0,
        temperature_ev=4.0,
        rng=3,
    )
    electrons.inject(source, 1e-9)
    v = electrons.velocities
    v_th = np.sqrt(4.0 * constants.e / ELECTRON_MASS)
    assert np.all(v[0] > 0.0)
    assert np.mean(v[0]) == pytest.approx(v_th * np.sqrt(np.pi / 2.0), rel=0.01)
    assert np.std(v[1]) == pytest.approx(v_th, rel=0.01)
    backwards = ReemissionSource(0.9, absorbed_weight=lambda: 100.0, direction=-1.0)
    other = Species(ndim=2, nvel=3)
    with pytest.raises(ValueError, match="y_range"):
        other.inject(backwards, 1e-9)
    line = ReemissionSource(
        0.2, absorbed_weight=lambda: 500.0, direction=-1.0, y_range=(0.25, 0.75)
    )
    assert other.inject(line, 1e-9) == 500
    x, y = other.positions
    assert np.all(x == 0.2) and y.min() >= 0.25 and y.max() <= 0.75
    assert np.all(other.velocities[0] < 0.0)


def test_reemission_checks() -> None:
    for kwargs in (
        {"net_current_density": -1.0},
        {"direction": 0.0},
        {"y_range": (1.0, 0.0)},
    ):
        with pytest.raises(ValueError):
            ReemissionSource(position=0.0, **kwargs)
    with pytest.raises(ValueError, match="point in 1-D"):
        ReemissionSource(position=0.0).emit(1.0, Species(ndim=3, nvel=3).info)


# Surfaces


PLATE = Polygon(vertices=[(0.4, 0.2), (0.6, 0.2), (0.6, 0.8), (0.4, 0.8)])
DOMAIN = ((0.0, 0.0), (0.03, 0.012))


def _plasma_grid() -> Aperture2D:
    return Aperture2D(
        axis=0,
        position=0.006,
        vertices=((0.008, 0.006), (0.008, 0.002), (0.011, 0.002), (0.011, 0.006)),
    )


def test_segments_have_outward_normals_for_either_orientation() -> None:
    for vertices in (
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 0)],
    ):
        segments = object_surface_segments(Polygon(vertices=vertices))
        assert len(segments) == 4
        for start, end, normal in segments:
            assert np.dot(normal, 0.5 * (start + end) - 0.5) > 0.0
    circle = object_surface_segments(Circle(midpoint=(0.0, 0.0), radius=1.0))
    assert len(circle) == 64
    with pytest.raises(TypeError):
        object_surface_segments(object())


def test_aperture_surface_excludes_domain_boundary_edges() -> None:
    everything = SurfaceSource(_plasma_grid(), current_density=1.0)
    source = SurfaceSource(_plasma_grid(), current_density=1.0, domain_bounds=DOMAIN)
    assert everything.segments[0].shape[0] == 8
    assert source.segments[0].shape[0] == 6
    assert source.total_length == pytest.approx(everything.total_length - 0.006)


def test_select_keeps_only_the_channel_walls() -> None:
    source = SurfaceSource(
        _plasma_grid(),
        current_density=1.0,
        select=lambda midpoint, normal: (
            abs(normal[1]) > 0.5 and 0.008 < midpoint[0] < 0.011
        ),
        domain_bounds=DOMAIN,
    )
    assert source.total_length == pytest.approx(0.006)
    positions, velocities = source.sample(400, thermal_velocity=1.0e4)
    assert np.all(np.abs(positions[1] - 0.006) < 0.002 + 1e-5)
    # emitted towards the channel axis, away from the wall
    assert np.all(np.sign(0.006 - positions[1]) == np.sign(velocities[1]))


def test_surface_emission_rate_and_markers_lie_just_outside() -> None:
    species = Species(ndim=2, nvel=2, weight=1.0e3)
    source = SurfaceSource(PLATE, current_density=2.0, rng=5, offset=1e-4)
    assert source.rate == pytest.approx(2.0 * 1.6 / constants.e)
    assert source.current_density == 2.0 and source.sampling is None
    dt = 1.0e-13
    created = sum(species.inject(source, dt) for _ in range(50))
    assert abs(created - source.rate * dt * 50 / 1.0e3) <= 1.0 + 1e-9
    positions, velocities = species.positions, species.velocities
    assert not any(PLATE.contains(tuple(p)) for p in positions.T)
    vertical = (positions[0] < 0.4) | (positions[0] > 0.6)
    assert np.all(
        np.sign(velocities[0, vertical]) == np.sign(positions[0, vertical] - 0.5)
    )
    assert np.all(
        np.sign(velocities[1, ~vertical]) == np.sign(positions[1, ~vertical] - 0.5)
    )


def test_surface_emission_temperature_overrides_the_species_temperature() -> None:
    species = Species(ndim=2, nvel=2, mass=PROTON_MASS)
    source = SurfaceSource(
        PLATE, current_density=constants.e * 1.0e5 / 1.6, temperature_ev=4.0, rng=2
    )
    species.inject(source, 1.0)
    v_th = np.sqrt(constants.e * 4.0 / PROTON_MASS)
    vertical = (species.positions[0] < 0.4) | (species.positions[0] > 0.6)
    assert np.std(species.velocities[1, vertical]) == pytest.approx(v_th, rel=0.05)


def test_surface_checks() -> None:
    with pytest.raises(ValueError, match="No emitting"):
        SurfaceSource(PLATE, current_density=1.0, select=lambda m, n: False)
    with pytest.raises(ValueError):
        SurfaceSource(PLATE, current_density=-1.0)
    with pytest.raises(ValueError):
        SurfaceSource(PLATE, current_density=1.0, offset=-1.0)
    source = SurfaceSource(PLATE, current_density=1.0)
    with pytest.raises(ValueError, match="cylindrical"):
        _ = source.total_area
    with pytest.raises(ValueError, match="2-D"):
        source.emit(1.0, Species(ndim=1, nvel=1).info)


def test_cylindrical_surface_emits_rings_with_an_azimuthal_velocity() -> None:
    disc = Polygon(vertices=[(0.0, 0.1), (0.1, 0.1), (0.1, 0.3), (0.0, 0.3)])
    source = SurfaceSource(disc, current_density=constants.e * 1.0e4, rng=4)
    source.cylindrical = True
    assert source.cylindrical
    # the swept areas: two annuli and two cylinders
    area = 2 * np.pi * (0.3**2 - 0.1**2) + 2 * np.pi * 0.1 * (0.1 + 0.3)
    assert source.total_area == pytest.approx(area)
    species = Species(ndim=2, nvel=3, mass=PROTON_MASS)
    created = species.inject(source, 1.0)
    assert created == int(np.floor(area * 1.0e4))
    assert np.std(species.velocities[2]) == pytest.approx(species.v_th, rel=0.1)
    source.cylindrical = False
    assert source.total_length == pytest.approx(0.6)
    sampled = SurfaceSource(disc, current_density=1.0, sampling=ImportanceSampling())
    sampled.cylindrical = True
    with pytest.raises(NotImplementedError):
        sampled.sample_weighted(4, 1.0)


def test_surface_source_with_sampling_preserves_the_emission() -> None:
    species = Species(ndim=2, nvel=2)
    bias = PositionBias(points=[0.2, 0.8], values=[0.2, 1.0], axis=1)
    source = SurfaceSource(
        PLATE,
        current_density=1.0,
        rng=5,
        sampling=ImportanceSampling(temperature_ratio=3.0, position_bias=bias),
    )
    assert species.inject(source, 2.0e4 / source.rate) == 20_000
    w, positions = species.weights, species.positions
    assert np.sum(w) == pytest.approx(2.0e4, rel=0.03)
    assert np.count_nonzero(positions[1] > 0.7) > 2 * np.count_nonzero(
        positions[1] < 0.3
    )
    plain = SurfaceSource(PLATE, current_density=1.0, sampling=ImportanceSampling())
    _, _, factors = plain.sample_weighted(10, 1.0)
    np.testing.assert_allclose(factors, 1.0)


def test_products_leave_the_nearest_face_along_its_normal() -> None:
    source = SurfaceConversionSource(
        _plasma_grid(), "ions", 0.5, domain_bounds=DOMAIN, offset=0.0, rng=1
    )
    assert source.rate == 0.0 and not source.shares_flux
    rng = np.random.default_rng(0)
    n = 4000
    impacts = np.vstack([np.full(n, 0.0081), rng.uniform(0.0002, 0.0018, n)])
    source.add_impacts(impacts, np.full(n, 2.0))
    source.add_impacts(np.zeros((2, 0)), np.zeros(0))
    species = Species(ndim=2, nvel=3, mass=PROTON_MASS)
    assert species.inject(source, 1e-10) == 4000
    assert source.absorbed == pytest.approx(8000.0)
    assert source.emitted == pytest.approx(4000.0)
    positions, velocities = species.positions, species.velocities
    np.testing.assert_allclose(positions[0], 0.008)
    assert np.all(velocities[0] < 0.0)
    v_th = species.v_th
    assert np.mean(-velocities[0]) == pytest.approx(np.sqrt(np.pi / 2) * v_th, rel=0.05)
    assert np.std(velocities[2]) == pytest.approx(v_th, rel=0.05)
    assert np.sum(source.emitted_per_segment) == pytest.approx(4000.0)
    assert species.inject(source, 1e-10) == 0  # nothing new hit


def test_fractional_products_carry_over_between_steps() -> None:
    source = SurfaceConversionSource(
        _plasma_grid(), "ions", 0.1, domain_bounds=DOMAIN, rng=2
    )
    species = Species(ndim=2, nvel=2)
    total = 0
    for _ in range(25):
        source.add_impacts(np.array([[0.0081], [0.001]]), np.array([1.0]))
        total += species.inject(source, 1e-10)
    assert total == 2
    with pytest.raises(ValueError):
        SurfaceConversionSource(_plasma_grid(), "ions", -1.0)
    with pytest.raises(ValueError, match="2-D"):
        source.emit(1.0, Species(ndim=1, nvel=1).info)


def test_the_nearest_point_is_on_a_selected_face() -> None:
    channel = SurfaceConversionSource(
        _plasma_grid(),
        "ions",
        1.0,
        domain_bounds=DOMAIN,
        select=lambda midpoint, normal: abs(normal[0]) < 0.5,
    )
    points, _ = channel.nearest_surface(np.array([[0.0081, 0.0105], [0.0035, 0.0039]]))
    np.testing.assert_allclose(points[:, 1], 0.004)
    np.testing.assert_allclose(points[:, 0], [0.0081, 0.0105])


# Directed source


MASS = 132.905 * constants.atomic_mass
T_EV = 0.05
AXIS = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)


def _directed(**options) -> DirectedSource:
    return DirectedSource(
        (0.1, 0.2, 0.3), AXIS, rate=1.0e15, temperature_ev=T_EV, **options
    )


@pytest.mark.parametrize("exponent", [0.0, 2.0, 5.0])
def test_cos_theta_follows_the_cosine_power(exponent: float) -> None:
    n = 200_000
    positions, velocities = _directed(exponent=exponent, rng=1).sample(n, MASS)
    positions, velocities = _host(positions), _host(velocities)
    np.testing.assert_allclose(positions, np.repeat([[0.1], [0.2], [0.3]], n, axis=1))
    mu = AXIS @ velocities / np.linalg.norm(velocities, axis=0)
    assert mu.min() >= 0.0
    counts, edges = np.histogram(mu, bins=20, range=(0, 1))
    expected = n * (edges[1:] ** (exponent + 1) - edges[:-1] ** (exponent + 1))
    keep = expected > 50
    chi2 = np.sum((counts - expected)[keep] ** 2 / expected[keep])
    assert chi2 / keep.sum() < 2.0


def test_directed_speeds_are_those_of_an_effusion_flux() -> None:
    v_th = np.sqrt(constants.e * T_EV / MASS)
    speeds = _directed(rng=2).speeds(400_000, MASS)
    assert speeds.mean() == pytest.approx(1.5 * np.sqrt(np.pi / 2) * v_th, rel=3e-3)
    assert np.mean(speeds**2) == pytest.approx(4 * v_th**2, rel=5e-3)


def test_intensity_is_normalised_and_the_table_reproduces_the_power() -> None:
    theta = np.linspace(0, np.pi / 2, 91)
    fine = np.linspace(0, np.pi / 2, 20001)
    for source in (
        _directed(exponent=2.0),
        _directed(angular_table=(theta, np.cos(theta) ** 2)),
    ):
        integrand = source.intensity(np.cos(fine)) * 2 * np.pi * np.sin(fine)
        total = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(fine))
        assert total == pytest.approx(1.0, rel=1e-4)
    table = _directed(angular_table=(theta, np.cos(theta) ** 2), rng=3)
    power = _directed(exponent=2.0)
    mu = np.linspace(0.05, 1.0, 20)
    np.testing.assert_allclose(
        table.intensity(mu), power.intensity(mu), rtol=2e-3, atol=1e-4
    )
    _, velocities = table.sample(100_000, MASS)
    velocities = _host(velocities)
    cosine = AXIS @ velocities / np.linalg.norm(velocities, axis=0)
    assert cosine.mean() == pytest.approx(3.0 / 4.0, abs=0.003)
    assert power.intensity(np.array([-0.5]))[0] == 0.0


def test_a_disc_spreads_the_positions_uniformly() -> None:
    radius = 0.01
    positions, _ = _directed(radius=radius, rng=4).sample(100_000, MASS)
    offset = _host(positions) - np.array([[0.1], [0.2], [0.3]])
    np.testing.assert_allclose(AXIS @ offset, 0.0, atol=1e-15)
    r = np.linalg.norm(offset, axis=0)
    assert r.max() <= radius
    assert np.mean(r**2) == pytest.approx(radius**2 / 2, rel=0.01)


def test_directed_emit_carries_the_remainder() -> None:
    species = Species(ndim=3, nvel=3, weight=1.0e9, mass=MASS)
    source = _directed(rng=5)
    assert source.centre == (0.1, 0.2, 0.3)
    np.testing.assert_allclose(source.axis, AXIS)
    assert source.temperature_ev == T_EV and source.rate == 1.0e15
    assert source.marker_weight(10) == 1.0e14
    source.flux_fraction = 0.5
    emitted = sum(species.inject(source, 1.0e-7) for _ in range(30))
    # 0.5 * 1e15 /s * 3e-6 s / 1e9 = 1.5 markers
    assert emitted == 1
    with pytest.raises(ValueError, match="3D"):
        source.emit(1.0, Species(ndim=2, nvel=3).info)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (((0, 0, 0), (0, 0, 0)), {}),
        (((0, 0, 0), (0, 0, 1)), {"rate": -1.0}),
        (((0, 0, 0), (0, 0, 1)), {"exponent": -1.0}),
        (((0, 0, 0), (0, 0, 1)), {"temperature_ev": 0.0}),
        (((0, 0, 0), (0, 0, 1)), {"radius": -1.0}),
        (((0, 0), (0, 0, 1)), {}),
        (((0, 0, 0), (0, 0, np.inf)), {}),
        (((0, 0, 0), (0, 0, 1)), {"angular_table": ([0.0, 2.0], [1.0, 1.0])}),
        (((0, 0, 0), (0, 0, 1)), {"angular_table": ([0.0], [1.0])}),
        (((0, 0, 0), (0, 0, 1)), {"angular_table": ([0.0, 1.0], [0.0, 0.0])}),
    ],
)
def test_directed_constructor_checks(args, kwargs) -> None:
    options = {"rate": 1.0, "temperature_ev": 1.0, **kwargs}
    with pytest.raises(ValueError):
        DirectedSource(*args, **options)
