"""Timestep sources: markers emitted from planes, volumes, surfaces and nozzles.

A source creates the markers of one time step. ``emit(dt, species)`` returns their
positions ``(ndim, n)``, velocities ``(nvel, n)`` and weights ``(n,)`` as arrays, and
``inject(store, dt, species)`` adds them to a `ParticleStore`. `MarkerSpecies` gives a
source what it needs of the species: charge, mass, marker weight, thermal speed and
the numbers of position and velocity components. Every source carries the fraction of
a marker it could not emit to the next step, so its long-time rate is exact, and
creates only ``flux_fraction`` of its rate (``1 / P`` when ``P`` ranks share it).

- `PlanarFluxInjector1D` and `PlanarFluxInjector2D`: a current density through a plane
  (a point in 1D, a line in 2D), with flux-weighted half-Maxwellian normal speeds;
  `planar_flux_injector` picks one for a box. Their ``sample_thermalized_velocities``
  gives fresh velocities to markers a code finds crossing the plane.
- `UniformVolumeSource` and `ProfiledVolumeSource`: a rate in a box, uniform or with a
  profile along one axis (`cosine_profile`).
- `ChargeBalanceSource` and `ReemissionSource`: emission that follows what a wall
  absorbed (a callable gives the cumulative amount).
- `SurfaceSource` and `SurfaceConversionSource`: emission from the boundary segments of
  a 2D object (`object_surface_segments`), at a rate or from recorded impacts.
- `DirectedSource`: a point or disc emitting about an axis, ``cos^n`` or tabulated.

Each source draws from its own generator, ``rng`` (a NumPy or CuPy generator, or a seed
for a new one), so a seeded run is reproducible. With ``sampling``
(`mpiparticles.sampling.ImportanceSampling`) the plane, volume and surface sources draw
from a sampling distribution and weight the markers by ``f / s``.

Examples
--------
>>> import numpy as np
>>> from mpiparticles import ParticleStore
>>> from mpiparticles.injection import MarkerSpecies, UniformVolumeSource
>>> from simgeometry.boundingbox import BoundingBox
>>> box = BoundingBox(lower_bound=(0.0,), upper_bound=(1.0,))
>>> source = UniformVolumeSource(box, rate=1.0e3, rng=np.random.default_rng(0))
>>> species = MarkerSpecies(charge=-1.6e-19, mass=9.1e-31, weight=10.0, ndim=1, nvel=3)
>>> store = ParticleStore(ndim=1, nvel=3)
>>> source.inject(store, 0.05, species)
5
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cunumpy as xp
import numpy as np
from methodcard import method
from scipy import constants
from simgeometry.boundingbox import BoundingBox
from simgeometry.cylindrical import radial_weight_factors

from mpiparticles._types import Array
from mpiparticles.sampling import (
    ImportanceSampling,
    as_generator,
    flux_speeds,
    normal,
    uniform,
)
from mpiparticles.store import ParticleStore

#: Points of the table the inverse CDF of a profiled source is interpolated from.
_PROFILE_TABLE_SIZE = 4097


@dataclass(frozen=True)
class MarkerSpecies:
    """What a source needs to know of the species it emits.

    Parameters
    ----------
    charge : float
        Charge of one physical particle [C].
    mass : float
        Mass of one physical particle [kg].
    weight : float
        Physical particles per marker.
    thermal_velocity : float
        ``sqrt(T / m)`` of the species [m/s]; sources with their own temperature
        ignore it.
    ndim : int
        Position components.
    nvel : int
        Velocity components.
    """

    charge: float
    mass: float
    weight: float
    thermal_velocity: float = 0.0
    ndim: int = 1
    nvel: int = 1

    def __post_init__(self) -> None:
        """Validate the species."""
        if not self.weight > 0.0:
            raise ValueError("weight must be > 0")
        if not self.mass > 0.0:
            raise ValueError("mass must be > 0")
        if not self.thermal_velocity >= 0.0:
            raise ValueError("thermal_velocity must be >= 0")

    def thermal_velocity_at(self, temperature_ev: float | None) -> float:
        """Return ``sqrt(e T / m)`` at ``temperature_ev``, or the species' own speed.

        Parameters
        ----------
        temperature_ev : float, optional
            A temperature in eV; None gives `thermal_velocity`.

        Returns
        -------
        float
            The thermal speed [m/s].
        """
        if temperature_ev is None:
            return float(self.thermal_velocity)
        return float(np.sqrt(constants.e * temperature_ev / self.mass))


def _empty(species: MarkerSpecies) -> tuple[Array, Array, Array]:
    """No markers, in the shapes of ``species``."""
    return (
        xp.empty((species.ndim, 0)),
        xp.empty((species.nvel, 0)),
        xp.empty(0),
    )


def _pad_velocities(velocities: Array, nvel: int) -> Array:
    """Append zero components up to ``nvel``."""
    missing = nvel - velocities.shape[0]
    if missing <= 0:
        return velocities
    extra = xp.zeros((missing, velocities.shape[1]))
    return xp.concatenate([xp.asarray(velocities), extra], axis=0)


class _Source:
    """The flux fraction and `inject` shared by the sources."""

    _flux_fraction: float = 1.0

    @property
    def flux_fraction(self) -> float:
        """Return the fraction of the rate this instance creates (1 unless split)."""
        return self._flux_fraction

    @flux_fraction.setter
    def flux_fraction(self, value: float) -> None:
        """Create only this fraction of the rate, e.g. 1/P when P ranks share it."""
        if not 0.0 < value <= 1.0:
            raise ValueError("flux_fraction must be in (0, 1]")
        self._flux_fraction = float(value)

    def emit(self, dt: float, species: MarkerSpecies) -> tuple[Array, Array, Array]:
        """Return the positions, velocities and weights of one step's markers.

        Parameters
        ----------
        dt : float
            The time step [s].
        species : MarkerSpecies
            The emitted species.

        Returns
        -------
        positions : array, shape (ndim, n)
            The positions.
        velocities : array, shape (nvel, n)
            The velocities.
        weights : array, shape (n,)
            The weights.
        """
        raise NotImplementedError  # pragma: no cover

    def inject(self, store: ParticleStore, dt: float, species: MarkerSpecies) -> int:
        """Add one step's markers (`emit`) to ``store``; return how many.

        Parameters
        ----------
        store : ParticleStore
            The store, of ``species.ndim`` and ``species.nvel`` components.
        dt : float
            The time step [s].
        species : MarkerSpecies
            The emitted species.

        Returns
        -------
        int
            The number of new markers.
        """
        if store.ndim != species.ndim or store.nvel != species.nvel:
            raise ValueError("the store and the species differ in their components")
        positions, velocities, weights = self.emit(dt, species)
        num = int(weights.shape[0])
        if num:
            store.add(
                xp.asarray(positions), xp.asarray(velocities), xp.asarray(weights)
            )
        return num


class _CountedSource(_Source):
    """A source at a fixed rate that carries its fractional marker."""

    _marker_remainder: float = 0.0

    def _count(self, exact_per_step: float) -> int:
        """Markers of this step for ``exact_per_step`` new ones, carrying the remainder."""
        exact = exact_per_step + self._marker_remainder
        num = int(np.floor(exact))
        self._marker_remainder = float(exact - num)
        return num


def _balanced(num: int, next_direction: float) -> tuple[Array, float]:
    """Alternating ``+-1`` starting at ``next_direction``, and the next start."""
    directions = xp.ones(num, dtype=float)
    directions[::2] = next_direction
    directions[1::2] = -next_direction
    return directions, (-next_direction if num % 2 == 1 else next_direction)


@method(
    equations={
        "markers": r"N^n = \left\lfloor \frac{J\,\Delta t\, f_P}{|q|\, w} + r^{n-1} \right\rfloor, \qquad r^n = \frac{J\,\Delta t\, f_P}{|q|\, w} + r^{n-1} - N^n",
        "speeds": r"v_n = v_\mathrm{th}\sqrt{-2\ln(1 - u)}, \quad u \sim U(0, 1), \qquad v_\mathrm{th} = \sqrt{T/m}",
    },
    assumptions=(
        r"one way, or in balanced pairs; $f_P = 1/P$ when $P$ ranks share the rate",
        r"the fractional marker $r$ is carried, so the long-time current is exact",
    ),
)
class PlanarFluxInjector1D(_CountedSource):
    """Inject one-dimensional markers from a plane at a target current density.

    The normal speeds come from the flux-weighted half-Maxwellian at the species
    temperature, towards ``direction`` (``None`` alternates both ways in balanced
    pairs). Velocity components after the first are zero. With ``sampling`` the speeds
    come from the flux distribution at ``sampling.temperature_ratio`` times the species
    temperature and the weights carry ``f / s`` (a position bias has no effect on a
    plane).

    Parameters
    ----------
    position : float
        The plane [m].
    current_density : float
        Magnitude of the current density [A/m^2].
    rng : generator, int or None
        The random generator, or a seed for a new one.
    direction : float, optional
        ``+1`` or ``-1``; None for balanced flux both ways.
    sampling : ImportanceSampling, optional
        Importance sampling of the injected markers.
    """

    def __init__(
        self,
        position: float,
        current_density: float,
        rng: Any = None,
        direction: float | None = None,
        sampling: ImportanceSampling | None = None,
    ) -> None:
        """Initialize the planar flux source."""
        if current_density < 0.0:
            raise ValueError("current_density must be a magnitude >= 0")
        if direction is not None and direction == 0.0:
            raise ValueError("direction must be non-zero when provided")
        self._position = float(position)
        self._current_density = float(current_density)
        self._direction = (
            None if direction is None else (1.0 if direction > 0.0 else -1.0)
        )
        self._sampling = sampling
        self._rng = as_generator(rng)
        self._next_direction = 1.0
        self._next_thermalization_direction = 1.0
        self._bounding_box = BoundingBox(
            lower_bound=(self._position,), upper_bound=(self._position,)
        )

    def _directions(self, num: int, thermalization: bool) -> Array:
        if self._direction is not None:
            return xp.full(num, self._direction, dtype=float)
        if thermalization:
            directions, self._next_thermalization_direction = _balanced(
                num, self._next_thermalization_direction
            )
        else:
            directions, self._next_direction = _balanced(num, self._next_direction)
        return directions

    @property
    def position(self) -> float:
        """Return the source-plane position."""
        return self._position

    @property
    def current_density(self) -> float:
        """Return the source current-density magnitude."""
        return self._current_density

    @property
    def direction(self) -> float | None:
        """Return the fixed injection direction, or None for balanced flux."""
        return self._direction

    @property
    def sampling(self) -> ImportanceSampling | None:
        """Return the importance sampling of injected markers, if any."""
        return self._sampling

    @property
    def bounding_box(self) -> BoundingBox:
        """Return the zero-thickness source-plane bounds."""
        return self._bounding_box

    def markers_per_step(self, charge: float, weight: float, dt: float) -> int:
        """Return the marker count of one step, carrying the fractional remainder.

        Parameters
        ----------
        charge : float
            Charge of a physical particle [C].
        weight : float
            Physical particles per marker.
        dt : float
            The time step [s].

        Returns
        -------
        int
            The number of markers.
        """
        if dt < 0.0:
            raise ValueError("dt must be >= 0")
        if weight <= 0.0:
            raise ValueError("weight must be > 0")
        if charge == 0.0:
            raise ValueError("charge must be non-zero")
        per_area = self._current_density * self._flux_fraction * dt / abs(charge)
        return self._count(per_area / weight)

    def emit(self, dt: float, species: MarkerSpecies) -> tuple[Array, Array, Array]:
        """Return one step's markers (see `_Source.emit`)."""
        if species.ndim != 1:
            raise ValueError("PlanarFluxInjector1D only supports 1-D species")
        num = self.markers_per_step(species.charge, species.weight, dt)
        if num == 0:
            return _empty(species)
        positions = xp.full((1, num), self._position, dtype=float)
        weights = xp.full(num, species.weight, dtype=float)
        directions = self._directions(num, thermalization=False)
        if self._sampling is None:
            speeds = flux_speeds(self._rng, num, species.thermal_velocity)
        else:
            speeds, factors = self._sampling.sample_flux_speeds(
                self._rng, num, species.thermal_velocity
            )
            weights = weights * factors
        velocities = (directions * speeds)[None, :]
        return positions, _pad_velocities(velocities, species.nvel), weights

    def sample_thermalized_velocities(
        self, num_markers: int, thermal_velocity: float
    ) -> Array:
        """Return ``(1, num_markers)`` source velocities for markers crossing the plane.

        Parameters
        ----------
        num_markers : int
            The number of markers.
        thermal_velocity : float
            The thermal speed [m/s].

        Returns
        -------
        array, shape (1, num_markers)
            The velocities.
        """
        if num_markers < 0:
            raise ValueError("num_markers must be >= 0")
        if thermal_velocity < 0.0:
            raise ValueError("thermal_velocity must be >= 0")
        if num_markers == 0:
            return xp.empty((1, 0), dtype=float)
        directions = self._directions(num_markers, thermalization=True)
        speeds = flux_speeds(self._rng, num_markers, thermal_velocity)
        return (directions * speeds)[None, :]


class PlanarFluxInjector2D(_CountedSource):
    """Inject 2-D markers from a line at ``x = x_source`` each time step.

    The rate is a current density *J* [A/m^2] integrated over the line, so marker
    weights are physical particles per metre of depth. Markers start at
    ``x = x_source`` with ``y`` uniform on ``[y_lo, y_hi]``. The normal velocity is
    drawn from the flux-weighted half-Maxwellian towards ``inject_direction`` (``None``
    alternates +x and -x in balanced pairs); the tangential one is a Maxwellian
    (``transverse="thermal"``) or zero (``"cold"``). In cylindrical ``(x, r)``
    coordinates (``cylindrical = True``) the line is a disc or annulus emitting rings,
    weighted by ``r``, and a third velocity component is the azimuthal one of a thermal
    ring.

    With ``rethermalize=True`` the line is also a thermalization plane: a code gives
    markers crossing it fresh velocities from ``sample_thermalized_velocities``.

    With ``sampling``, velocities come from the source distribution at
    ``sampling.temperature_ratio`` times the species temperature, ``y`` is biased by
    ``sampling.position_bias``, and the weights carry ``f / s``.

    Parameters
    ----------
    x_source : float
        x-coordinate of the injection line [m].
    y_lo, y_hi : float
        y-extent of the injection line [m].
    current_density : float
        Magnitude of the current density [A/m^2].
    inject_direction : float, optional
        +1 to inject towards +x, -1 towards -x, None both ways with equal flux.
    rng : generator, int or None
        The random generator, or a seed for a new one.
    transverse : str
        ``"thermal"`` (Maxwellian v_y) or ``"cold"`` (v_y = 0).
    rethermalize : bool
        Whether a code resamples the velocities of markers crossing the line.
    sampling : ImportanceSampling, optional
        Importance sampling of the injected markers.
    """

    #: Axisymmetric (x, r) source: a disc or annulus at x = x_source, emitting rings.
    cylindrical = False

    def __init__(
        self,
        x_source: float,
        y_lo: float,
        y_hi: float,
        current_density: float,
        inject_direction: float | None = 1.0,
        rng: Any = None,
        *,
        transverse: str = "thermal",
        rethermalize: bool = False,
        sampling: ImportanceSampling | None = None,
    ) -> None:
        """Initialize the 2-D planar flux source."""
        if current_density < 0.0:
            raise ValueError("current_density must be >= 0")
        if y_hi <= y_lo:
            raise ValueError("y_hi must be > y_lo")
        if inject_direction is not None and inject_direction == 0.0:
            raise ValueError("inject_direction must be non-zero when provided")
        if transverse not in ("thermal", "cold"):
            raise ValueError("transverse must be 'thermal' or 'cold'")
        self._x_source = float(x_source)
        self._y_lo = float(y_lo)
        self._y_hi = float(y_hi)
        self._current_density = float(current_density)
        self._inject_direction = (
            None
            if inject_direction is None
            else (1.0 if inject_direction > 0.0 else -1.0)
        )
        self._transverse = transverse
        self.rethermalizes = bool(rethermalize)
        self._sampling = sampling
        self._mean_bias: float | None = None
        self._rng = as_generator(rng)
        self._next_direction = 1.0
        self._next_thermalization_direction = 1.0
        self._bounding_box = BoundingBox(
            lower_bound=(self._x_source, self._y_lo),
            upper_bound=(self._x_source, self._y_hi),
        )

    def _directions(self, num: int, thermalization: bool) -> Array:
        if self._inject_direction is not None:
            return xp.full(num, self._inject_direction, dtype=float)
        if thermalization:
            directions, self._next_thermalization_direction = _balanced(
                num, self._next_thermalization_direction
            )
        else:
            directions, self._next_direction = _balanced(num, self._next_direction)
        return directions

    def _velocities(self, num: int, v_th: float, thermalization: bool) -> Array:
        vx = self._directions(num, thermalization) * flux_speeds(self._rng, num, v_th)
        if self._transverse == "cold":
            vy = xp.zeros(num)
        else:
            vy = normal(self._rng, 0.0, v_th, num)
        return xp.stack([vx, vy], axis=0)

    def _draw_positions(self, num: int) -> Array:
        """Positions ``(2, num)`` on the source line: fixed x, uniform y (or r)."""
        x_pos = xp.full(num, self._x_source)
        y_pos = uniform(self._rng, self._y_lo, self._y_hi, num)
        return xp.stack([x_pos, y_pos], axis=0)

    def _sample_biased(self, num: int, v_th: float) -> tuple[Array, Array, Array]:
        """Positions, velocities and weight factors from the importance sampling."""
        sampling = self._sampling
        assert sampling is not None
        if self._mean_bias is None:
            self._mean_bias = sampling.mean_bias(
                self._draw_positions,
                self._bounding_box.lower_bound,
                self._bounding_box.upper_bound,
            )
        positions, factors = sampling.sample_positions(
            self._rng, num, self._draw_positions, self._mean_bias
        )
        speeds, normal_factors = sampling.sample_flux_speeds(self._rng, num, v_th)
        vx = self._directions(num, thermalization=False) * speeds
        factors = factors * normal_factors
        if self._transverse == "cold":
            vy = xp.zeros(num)
        else:
            vy, transverse_factors = sampling.sample_transverse(self._rng, num, v_th)
            factors = factors * transverse_factors
        return positions, xp.stack([vx, vy], axis=0), factors

    @property
    def current_density(self) -> float:
        """Return the source current-density magnitude [A/m^2]."""
        return self._current_density

    @property
    def bounding_box(self) -> BoundingBox:
        """Return the zero-thickness source line bounds."""
        return self._bounding_box

    @property
    def position(self) -> float:
        """Return the thermalization plane ``x = x_source``."""
        return self._x_source

    @property
    def direction(self) -> float | None:
        """Return the fixed injection direction, or None for balanced +-x."""
        return self._inject_direction

    @property
    def sampling(self) -> ImportanceSampling | None:
        """Return the importance sampling of injected markers, if any."""
        return self._sampling

    def markers_per_step(self, charge: float, weight: float, dt: float) -> int:
        """Return the marker count of one step, carrying the fractional remainder.

        Parameters
        ----------
        charge : float
            Charge of a physical particle [C].
        weight : float
            Physical particles per marker.
        dt : float
            The time step [s].

        Returns
        -------
        int
            The number of markers.
        """
        per_area = self._current_density * self._flux_fraction * dt / abs(charge)
        # the line's length per unit depth, or the area of the annulus
        extent = (
            np.pi * (self._y_hi**2 - self._y_lo**2)
            if self.cylindrical
            else self._y_hi - self._y_lo
        )
        return self._count(per_area * extent / weight)

    def emit(self, dt: float, species: MarkerSpecies) -> tuple[Array, Array, Array]:
        """Return one step's markers (see `_Source.emit`)."""
        if species.ndim != 2:
            raise ValueError("PlanarFluxInjector2D only supports 2-D species")
        num = self.markers_per_step(species.charge, species.weight, dt)
        if num == 0:
            return _empty(species)
        v_th = species.thermal_velocity
        weights = xp.full(num, species.weight)
        if self._sampling is None:
            positions = self._draw_positions(num)
            velocities = self._velocities(num, v_th, thermalization=False)
            if self.cylindrical:
                # rings uniform in r, weighted ~ r: a uniform current density
                weights = weights * radial_weight_factors(
                    positions[1], self._y_lo, self._y_hi
                )
        else:
            if self.cylindrical:
                raise NotImplementedError(
                    "importance sampling is not available in cylindrical coordinates"
                )
            positions, velocities, factors = self._sample_biased(num, v_th)
            weights = weights * factors
        if species.nvel > 2:
            extra = xp.zeros((species.nvel - 2, num))
            if self.cylindrical and self._transverse != "cold":
                # the azimuthal velocity of a thermal ring
                extra[0] = normal(self._rng, 0.0, v_th, num)
            velocities = xp.concatenate([velocities, extra], axis=0)
        return positions, velocities, weights

    def sample_thermalized_velocities(
        self, num_markers: int, thermal_velocity: float
    ) -> Array:
        """Return source velocities for markers crossing the line.

        Parameters
        ----------
        num_markers : int
            The number of markers.
        thermal_velocity : float
            The thermal speed [m/s].

        Returns
        -------
        array, shape (2, num_markers) or (3, num_markers)
            The velocities; with the azimuthal one in cylindrical coordinates.
        """
        if num_markers < 0:
            raise ValueError("num_markers must be >= 0")
        if thermal_velocity < 0.0:
            raise ValueError("thermal_velocity must be >= 0")
        velocities = self._velocities(
            num_markers, thermal_velocity, thermalization=True
        )
        if self.cylindrical and self._transverse != "cold":
            azimuthal = normal(self._rng, 0.0, thermal_velocity, num_markers)
            velocities = xp.concatenate([velocities, azimuthal[None, :]], axis=0)
        return velocities


def planar_flux_injector(
    box: BoundingBox,
    current_density: float,
    sampling: ImportanceSampling | None = None,
    rng: Any = None,
) -> PlanarFluxInjector1D | PlanarFluxInjector2D:
    """Return the planar flux source on the lower ``x`` face of ``box``.

    Parameters
    ----------
    box : BoundingBox
        A 1-D or 2-D box; in 2-D the line spans its ``y`` extent.
    current_density : float
        The current density [A/m^2] emitted from ``x = box.lower_bound[0]``.
    sampling : ImportanceSampling, optional
        Importance sampling of the injected markers.
    rng : generator, int or None
        The random generator, or a seed for a new one.

    Returns
    -------
    PlanarFluxInjector1D or PlanarFluxInjector2D
        The source.
    """
    lower, upper = box.lower_bound, box.upper_bound
    if len(lower) == 1:
        return PlanarFluxInjector1D(
            position=lower[0],
            current_density=current_density,
            rng=rng,
            sampling=sampling,
        )
    if len(lower) == 2:
        return PlanarFluxInjector2D(
            x_source=lower[0],
            y_lo=lower[1],
            y_hi=upper[1],
            current_density=current_density,
            rng=rng,
            sampling=sampling,
        )
    raise NotImplementedError("planar-flux injection is only supported in 1-D and 2-D.")


@method(
    equations={
        "markers": r"N^n = \left\lfloor \frac{R\,\Delta t\, f_P}{w} + r^{n-1} \right\rfloor",
        "velocities": r"\mathbf{v} \sim \mathcal{N}(0, v_\mathrm{th}^2 I), \qquad v_\mathrm{th} = \sqrt{T/m}",
    },
    assumptions=(
        r"$R$ physical particles per second per unit of the dimensions the box does not span, uniform in the box (or along one axis with a profile, `ProfiledVolumeSource`, e.g. the cosine of Charoy et al.)",
        r"a shared position generator (`pair_rng`) puts electrons and ions at the same points: neutral pairs",
    ),
    references=(
        "[Charoy et al., Plasma Sources Sci. Technol. 28, 105010 (2019)](https://doi.org/10.1088/1361-6595/ab46c5)",
    ),
)
class UniformVolumeSource(_CountedSource):
    """Create markers at a fixed rate, uniformly inside a box, each time step.

    The simplest model of volume ionization: ``rate`` physical particles per second
    per unit of the dimensions the box does not span (per metre of depth in 2-D, per
    square metre in 1-D), uniform over the box with Maxwellian velocities at the
    species temperature. In cylindrical ``(x, r)`` coordinates (``cylindrical = True``)
    ``rate`` is the total in the annulus and the rings are weighted by ``r``.

    Parameters
    ----------
    bounding_box : BoundingBox
        Region of creation (its dimension is the species dimension).
    rate : float
        Physical particles created per second (per unit transverse extent).
    rng : generator, int or None
        The random generator, or a seed for a new one.
    sampling : ImportanceSampling, optional
        Importance sampling: markers are drawn from its sampling distribution and
        carry ``f / s`` in their weight.
    """

    #: Axisymmetric (x, r) source: ``rate`` is the total rate [1/s] in the annulus.
    cylindrical = False

    def __init__(
        self,
        bounding_box: BoundingBox,
        rate: float,
        rng: Any = None,
        sampling: ImportanceSampling | None = None,
    ) -> None:
        """Initialize the volume source."""
        if rate < 0.0:
            raise ValueError("rate must be >= 0")
        self._bounding_box = bounding_box
        self._sampling = sampling
        self._mean_bias: float | None = None
        self._rate = float(rate)
        self._rng = as_generator(rng)

    def _draw_positions(self, num: int) -> Array:
        """Uniform positions ``(ndim, num)`` in the creation box."""
        ndim = self._bounding_box.ndim
        lower = xp.asarray(self._bounding_box.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self._bounding_box.upper_bound, dtype=float)[:, None]
        return lower + (upper - lower) * self._rng.random(size=(ndim, num))

    @property
    def bounding_box(self) -> BoundingBox:
        """Return the creation region."""
        return self._bounding_box

    @property
    def rate(self) -> float:
        """Return the physical creation rate [1/s per unit transverse extent]."""
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        """Change the rate from the next step on, e.g. from a density controller."""
        if not value >= 0.0:  # also rejects NaN
            raise ValueError("rate must be >= 0")
        self._rate = float(value)

    @property
    def sampling(self) -> ImportanceSampling | None:
        """Return the importance sampling of created markers, if any."""
        return self._sampling

    def markers_per_step(self, weight: float, dt: float) -> int:
        """Return the marker count of one step, carrying the fractional remainder.

        Parameters
        ----------
        weight : float
            Physical particles per marker.
        dt : float
            The time step [s].

        Returns
        -------
        int
            The number of markers.
        """
        return self._count(self._rate * self._flux_fraction * dt / weight)

    def emit(self, dt: float, species: MarkerSpecies) -> tuple[Array, Array, Array]:
        """Return one step's markers (see `_Source.emit`)."""
        ndim = self._bounding_box.ndim
        if species.ndim != ndim:
            raise ValueError(
                f"UniformVolumeSource box is {ndim}-D but the species is {species.ndim}-D",
            )
        num = self.markers_per_step(species.weight, dt)
        if num == 0:
            return _empty(species)
        nvel = species.nvel
        weights = xp.full(num, species.weight)
        if self._sampling is None:
            positions = self._draw_positions(num)
            velocities = normal(
                self._rng, 0.0, species.thermal_velocity, size=(nvel, num)
            )
            if self.cylindrical:
                # rings uniform in r, weighted ~ r: a uniform creation rate per volume
                box = self._bounding_box
                weights = weights * radial_weight_factors(
                    positions[1], box.lower_bound[1], box.upper_bound[1]
                )
        else:
            if self.cylindrical:
                raise NotImplementedError(
                    "importance sampling is not available in cylindrical coordinates"
                )
            if self._mean_bias is None:
                self._mean_bias = self._sampling.mean_bias(
                    self._draw_positions,
                    self._bounding_box.lower_bound,
                    self._bounding_box.upper_bound,
                )
            positions, position_factors = self._sampling.sample_positions(
                self._rng, num, self._draw_positions, self._mean_bias
            )
            velocities, velocity_factors = self._sampling.sample_maxwellian(
                self._rng, num, nvel, species.thermal_velocity
            )
            weights = weights * position_factors * velocity_factors
        return positions, velocities, weights


class ProfiledVolumeSource(UniformVolumeSource):
    """Create markers at a fixed rate with a density profile along one axis.

    Like `UniformVolumeSource`, but the creation rate per unit volume follows
    ``values`` at ``points`` (linear in between, zero outside) along ``axis`` and is
    uniform across the others. ``rate`` is the total over the box, so the profile only
    sets the shape: the ionization source of an axial-azimuthal ExB discharge is a
    cosine in x, uniform in y (`cosine_profile`).

    Sources whose ``pair_rng`` generators are seeded alike draw their positions from
    identical streams, so a pair of sources for an electron and an ion species creates
    the two at the same positions (as ionization does) as long as both make the same
    number of markers per step. The velocities come from ``rng`` and differ.

    Parameters
    ----------
    bounding_box : BoundingBox
        Region of creation.
    rate : float
        Physical particles created per second, the total over the box.
    points, values : array
        The profile: increasing points along ``axis`` and values ``>= 0``.
    axis : int
        The axis of the profile.
    rng : generator, int or None
        The random generator, or a seed for a new one.
    pair_rng : generator, int or None
        The generator of the positions; ``rng`` if None.
    sampling : ImportanceSampling, optional
        Importance sampling of the created markers.
    """

    def __init__(
        self,
        bounding_box: BoundingBox,
        rate: float,
        points: Array,
        values: Array,
        axis: int = 0,
        rng: Any = None,
        pair_rng: Any = None,
        sampling: ImportanceSampling | None = None,
    ) -> None:
        """Initialize the profiled source; the table is normalized to a CDF."""
        super().__init__(bounding_box, rate, rng=rng, sampling=sampling)
        points = np.asarray(points, dtype=float)
        values = np.asarray(values, dtype=float)
        if points.ndim != 1 or points.shape != values.shape or points.size < 2:
            raise ValueError("profile needs matching 1-D points and values (>= 2)")
        if np.any(np.diff(points) <= 0.0) or np.any(values < 0.0):
            raise ValueError("profile points must increase and values be >= 0")
        lower = bounding_box.lower_bound[axis]
        upper = bounding_box.upper_bound[axis]
        # the profile outside the box is not created
        fine = np.linspace(
            max(points[0], lower), min(points[-1], upper), _PROFILE_TABLE_SIZE
        )
        density = np.interp(fine, points, values)
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fine))]
        )
        if cdf[-1] <= 0.0:
            raise ValueError("profile has no weight inside the box")
        self._axis = int(axis)
        self._table_x = xp.asarray(fine)
        self._table_cdf = xp.asarray(cdf / cdf[-1])
        self._position_rng = self._rng if pair_rng is None else as_generator(pair_rng)

    def _draw_positions(self, num: int) -> Array:
        """Positions ``(ndim, num)``: the inverse CDF along ``axis``, uniform elsewhere."""
        ndim = self._bounding_box.ndim
        lower = xp.asarray(self._bounding_box.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self._bounding_box.upper_bound, dtype=float)[:, None]
        draws = self._position_rng.random(size=(ndim, num))
        positions = lower + (upper - lower) * draws
        positions[self._axis] = xp.interp(
            draws[self._axis], self._table_cdf, self._table_x
        )
        return positions


def cosine_profile(
    lower: float, upper: float, points: int = 257
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points, values)`` of a cosine on ``[lower, upper]``, zero at both ends.

    The shape of the ionization source of the Charoy et al. benchmark,
    ``cos(pi (x - x_m) / (upper - lower))`` about the middle ``x_m``.

    Parameters
    ----------
    lower, upper : float
        The interval.
    points : int
        The number of points.

    Returns
    -------
    tuple of numpy.ndarray
        The points and the values.
    """
    x = np.linspace(lower, upper, points)
    return x, np.maximum(
        np.cos(np.pi * (x - 0.5 * (lower + upper)) / (upper - lower)), 0.0
    )


@method(
    equations={
        "emitted": r"N_\mathrm{emitted}(t) = \frac{Q_\mathrm{absorbed}(t)}{|q|\, w}, \qquad \text{e.g. } N_{e,\mathrm{cathode}} = N_{e,\mathrm{anode}} - N_{i,\mathrm{anode}}",
    },
    assumptions=(
        r"follows the cumulative absorbed charge, so a step without net absorption is made up later",
        r"a full Maxwellian at `temperature_ev`, uniform in $y$ on the plane",
    ),
    references=(
        "[Charoy et al., Plasma Sources Sci. Technol. 28, 105010 (2019)](https://doi.org/10.1088/1361-6595/ab46c5)",
    ),
)
class ChargeBalanceSource(_Source):
    """Emit particles from a plane so that a wall absorbs no net charge.

    A discharge between a driven anode and a cathode that is not resolved (a plane
    inside the domain stands for it) needs the electrons the cathode emits to carry the
    current the anode collects. This source emits, at ``position`` along x and uniformly
    in y, as many particles as it takes to carry away the net charge absorbed at a wall,
    ``absorbed_charge()`` [C per unit depth, cumulative]: for electrons
    ``N_e,anode - N_i,anode`` per step, the "current continuity" emission of Charoy et
    al. (2019). A step in which the wall absorbs net positive charge is made up later.

    Parameters
    ----------
    position : float
        x of the emission plane [m].
    y_range : tuple of float
        The y extent of the plane [m].
    absorbed_charge : callable
        The cumulative net charge absorbed [C per unit depth].
    temperature_ev : float, optional
        Temperature of the emitted Maxwellian [eV]; the species' when None.
    rng : generator, int or None
        The random generator, or a seed for a new one.
    """

    def __init__(
        self,
        position: float,
        y_range: tuple[float, float],
        absorbed_charge: Callable[[], float],
        temperature_ev: float | None = None,
        rng: Any = None,
    ) -> None:
        """Initialize the emission plane."""
        if y_range[1] <= y_range[0]:
            raise ValueError("y_range must increase")
        self._position = float(position)
        self._y_range = (float(y_range[0]), float(y_range[1]))
        self._absorbed_charge = absorbed_charge
        self._temperature_ev = temperature_ev
        self._rng = as_generator(rng)
        self._emitted_weight = 0.0

    @property
    def position(self) -> float:
        """Return the x position of the emission plane."""
        return self._position

    @property
    def emitted_weight(self) -> float:
        """Return the physical particles emitted so far [per unit depth]."""
        return self._emitted_weight

    def emit(self, dt: float, species: MarkerSpecies) -> tuple[Array, Array, Array]:
        """Return the markers that carry the charge absorbed since the last call."""
        if species.ndim != 2:
            raise ValueError("ChargeBalanceSource is a line in a 2-D domain")
        wanted = self._absorbed_charge() / species.charge * self._flux_fraction
        weight = species.weight
        num = int(np.floor((wanted - self._emitted_weight) / weight))
        if num <= 0:
            return _empty(species)
        self._emitted_weight += num * weight
        v_th = species.thermal_velocity_at(self._temperature_ev)
        positions = xp.empty((2, num))
        positions[0] = self._position
        low, high = self._y_range
        positions[1] = low + (high - low) * self._rng.random(size=num)
        velocities = normal(self._rng, 0.0, v_th, size=(species.nvel, num))
        return positions, velocities, xp.full(num, weight)


@method(
    equations={
        "emitted": r"\frac{dN_\mathrm{emitted}}{dt} = \frac{dN_\mathrm{absorbed}}{dt} + \frac{J_\mathrm{net}}{|q|}",
        "speeds": r"v_n = v_\mathrm{th}\sqrt{-2\ln(1 - u)}, \qquad v_t \sim \mathcal{N}(0, v_\mathrm{th}^2)",
    },
    assumptions=(
        r"the cumulative count is followed, a fraction of a marker carried",
        r"the flux of a Maxwellian at `temperature_ev` along `direction`",
    ),
    references=(
        "[Boeuf and Chaudhury, Phys. Rev. Lett. 111, 155005 (2013)](https://doi.org/10.1103/PhysRevLett.111.155005)",
        "[Boeuf, arXiv:2211.17051 (2022)](https://arxiv.org/abs/2211.17051)",
    ),
)
class ReemissionSource(_Source):
    """Emit from a wall plane every particle the wall absorbs, plus a net current.

    The "net" cathode of Boeuf and Chaudhury (2013): the cathode emits a fixed net
    current density, and every particle that returns to it is emitted again. This source
    emits, at ``position`` along x, as many particles as the wall has absorbed
    (``absorbed_weight()``, cumulative physical particles per unit area or depth) plus
    ``net_current_density / |q|`` per second.

    The normal velocity component is drawn from the flux distribution of a Maxwellian
    at ``temperature_ev`` and points along ``direction``; the others are Gaussian. In
    2-D the plane spans ``y_range``; in 1-D it is a point.

    Parameters
    ----------
    position : float
        x of the emission plane [m].
    absorbed_weight : callable, optional
        The cumulative physical particles absorbed.
    net_current_density : float
        The net emitted current density magnitude [A/m^2].
    direction : float
        +1 or -1.
    temperature_ev : float, optional
        Temperature of the emission [eV]; the species' when None.
    y_range : tuple of float, optional
        The y extent of a 2-D plane [m].
    rng : generator, int or None
        The random generator, or a seed for a new one.
    """

    def __init__(
        self,
        position: float,
        absorbed_weight: Callable[[], float] | None = None,
        net_current_density: float = 0.0,
        direction: float = 1.0,
        temperature_ev: float | None = None,
        y_range: tuple[float, float] | None = None,
        rng: Any = None,
    ) -> None:
        """Initialize the emission plane."""
        if net_current_density < 0.0:
            raise ValueError("net_current_density must be a magnitude >= 0")
        if direction == 0.0:
            raise ValueError("direction must be +1 or -1")
        if y_range is not None and y_range[1] <= y_range[0]:
            raise ValueError("y_range must increase")
        self._position = float(position)
        self._absorbed_weight = absorbed_weight
        self._net_current_density = float(net_current_density)
        self._direction = 1.0 if direction > 0.0 else -1.0
        self._temperature_ev = temperature_ev
        self._y_range = (
            None if y_range is None else (float(y_range[0]), float(y_range[1]))
        )
        self._rng = as_generator(rng)
        self._emitted_weight = 0.0
        self._elapsed = 0.0

    @property
    def position(self) -> float:
        """Return the x position of the emission plane."""
        return self._position

    @property
    def emitted_weight(self) -> float:
        """Return the physical particles emitted so far [per unit area or depth]."""
        return self._emitted_weight

    @property
    def net_current_density(self) -> float:
        """Return the net emitted current density magnitude [A/m^2]."""
        return self._net_current_density

    def emit(self, dt: float, species: MarkerSpecies) -> tuple[Array, Array, Array]:
        """Return the markers absorbed so far and the net emission of this step."""
        ndim = species.ndim
        if ndim not in (1, 2):
            raise ValueError("ReemissionSource is a point in 1-D or a line in 2-D")
        if ndim == 2 and self._y_range is None:
            raise ValueError("a 2-D ReemissionSource needs a y_range")
        self._elapsed += dt
        absorbed = 0.0 if self._absorbed_weight is None else self._absorbed_weight()
        wanted = (
            absorbed + self._net_current_density / abs(species.charge) * self._elapsed
        ) * self._flux_fraction
        weight = species.weight
        num = int(np.floor((wanted - self._emitted_weight) / weight))
        if num <= 0:
            return _empty(species)
        self._emitted_weight += num * weight
        v_th = species.thermal_velocity_at(self._temperature_ev)
        positions = xp.empty((ndim, num))
        positions[0] = self._position
        if self._y_range is not None and ndim == 2:
            low, high = self._y_range
            positions[1] = low + (high - low) * self._rng.random(size=num)
        velocities = normal(self._rng, 0.0, v_th, size=(species.nvel, num))
        # the flux distribution of the normal component
        velocities[0] = self._direction * flux_speeds(self._rng, num, v_th)
        return positions, velocities, xp.full(num, weight)


def _polygon_segments(vertices: Array) -> list[tuple[Array, Array, Array]]:
    """Return ``(start, end, outward_normal)`` for every edge of a 2-D polygon."""
    points = np.asarray(vertices, dtype=float)
    n = points.shape[0]
    # the shoelace sign gives the orientation, which fixes the outward side
    area2 = sum(
        points[i, 0] * points[(i + 1) % n, 1] - points[(i + 1) % n, 0] * points[i, 1]
        for i in range(n)
    )
    orientation = 1.0 if area2 > 0.0 else -1.0
    segments = []
    for i in range(n):
        start = points[i]
        end = points[(i + 1) % n]
        edge = end - start
        length = float(np.hypot(edge[0], edge[1]))
        if length == 0.0:
            continue
        # for a counter-clockwise polygon the outward normal is the edge turned clockwise
        normal_vector = orientation * np.array([edge[1], -edge[0]]) / length
        segments.append((start, end, normal_vector))
    return segments


def _object_surface_segments(obj: Any) -> list[tuple[Array, Array, Array]]:
    """Return the boundary segments of the plain geometry of ``obj``."""
    if hasattr(obj, "_lower_polygon") and hasattr(obj, "_upper_polygon"):
        return _object_surface_segments(obj._lower_polygon) + _object_surface_segments(
            obj._upper_polygon
        )
    if hasattr(obj, "vertices") and getattr(obj, "closed", True):
        return _polygon_segments(np.asarray(obj.vertices, dtype=float))
    if hasattr(obj, "midpoint") and hasattr(obj, "radius"):
        centre = np.asarray(obj.midpoint, dtype=float)
        radius = float(obj.radius)
        angles = np.linspace(0.0, 2.0 * np.pi, 65)
        ring = centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        return _polygon_segments(ring[:-1])
    raise TypeError(f"Cannot sample the surface of {type(obj).__name__}")


def object_surface_segments(obj: Any) -> list[tuple[Array, Array, Array]]:
    """Return the straight boundary segments of a 2-D object with outward normals.

    Supports polygons (``vertices``), apertures (two polygons) and circles
    (``midpoint``/``radius``, approximated by 64 chords), such as simgeometry's
    objects. The normals of an inverted object point into its geometry, which is
    where its outside is.

    Parameters
    ----------
    obj : object
        The object.

    Returns
    -------
    list of tuple
        ``(start, end, outward_normal)`` of every segment.
    """
    segments = _object_surface_segments(obj)
    if getattr(obj, "inverted", False):
        segments = [(start, end, -normal) for start, end, normal in segments]
    return segments


@method(
    equations={
        "markers": r"N^n = \left\lfloor \frac{J\, L_\mathrm{segments}\,\Delta t}{e\, w} + r^{n-1} \right\rfloor",
        "velocities": r"v_n = v_\mathrm{th}\sqrt{-2\ln(1 - u)} \text{ along the outward normal}, \qquad v_t \sim \mathcal{N}(0, v_\mathrm{th}^2)",
    },
    assumptions=(
        r"2D only; uniform over the selected segments, starting `offset` outside the surface",
        r"`SurfaceConversionSource`: each absorbed projectile leaves as `conversion_yield` particles from the nearest segment (surface H$^-$ production)",
    ),
)
class SurfaceSource(_CountedSource):
    """Emit markers from the surface of a 2-D object into the domain each step.

    Models particles produced on a wall, such as negative ions formed on a caesiated
    plasma grid, or secondary electrons. ``current_density`` [A/m^2] is emitted
    uniformly over the selected boundary segments of ``obj``; markers start ``offset``
    outside the surface with a flux-weighted half-Maxwellian velocity along the outward
    normal and a Maxwellian tangential velocity, at ``temperature_ev`` (the species
    temperature when None). In cylindrical ``(x, r)`` coordinates (``cylindrical``) the
    surface is one of revolution about the x axis.

    ``select(midpoint, normal)`` filters the segments. By default every segment emits
    except those on the boundary of ``domain_bounds`` (an object cut by a periodic or
    symmetry face continues on the other side, so that edge is not a real surface).

    With ``sampling``, velocities come from the emission distribution at
    ``sampling.temperature_ratio`` times the emission temperature, the emission point is
    biased by ``sampling.position_bias``, and the weights carry ``f / s``.

    The draws are on the host (a NumPy generator).

    Parameters
    ----------
    obj : object
        The 2-D object (`object_surface_segments`).
    current_density : float
        The emitted current density [A/m^2].
    temperature_ev : float, optional
        The emission temperature [eV].
    select : callable, optional
        ``select(midpoint, normal)`` is True for emitting segments.
    domain_bounds : tuple, optional
        ``(lower, upper)`` of the domain.
    offset : float
        Distance of the emission points outside the surface [m].
    rng : generator, int or None
        A NumPy generator, or a seed for a new one.
    sampling : ImportanceSampling, optional
        Importance sampling of the emitted markers.
    """

    def __init__(
        self,
        obj: object,
        current_density: float,
        *,
        temperature_ev: float | None = None,
        select: Callable[[Array, Array], bool] | None = None,
        domain_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
        offset: float = 1.0e-6,
        rng: Any = None,
        sampling: ImportanceSampling | None = None,
    ) -> None:
        """Initialize the surface source."""
        if current_density < 0.0:
            raise ValueError("current_density must be >= 0")
        if offset < 0.0:
            raise ValueError("offset must be >= 0")
        segments = object_surface_segments(obj)
        if domain_bounds is not None:
            lower = np.asarray(domain_bounds[0], dtype=float)
            upper = np.asarray(domain_bounds[1], dtype=float)
            tolerance = 1.0e-9 * float(np.max(upper - lower))

            def on_domain_boundary(start: Array, end: Array) -> bool:
                for dim in range(2):
                    for face in (lower[dim], upper[dim]):
                        if (
                            abs(start[dim] - face) < tolerance
                            and abs(end[dim] - face) < tolerance
                        ):
                            return True
                return False

            segments = [
                seg for seg in segments if not on_domain_boundary(seg[0], seg[1])
            ]
        if select is not None:
            segments = [
                (start, end, normal_vector)
                for start, end, normal_vector in segments
                if select(0.5 * (start + end), normal_vector)
            ]
        if not segments:
            raise ValueError("No emitting surface segments were selected")
        self._starts = np.array([seg[0] for seg in segments])
        self._ends = np.array([seg[1] for seg in segments])
        self._normals = np.array([seg[2] for seg in segments])
        lengths = np.hypot(*(self._ends - self._starts).T)
        self._lengths = lengths
        self._total_length = float(lengths.sum())
        self._cumulative = np.cumsum(lengths) / self._total_length
        self._cylindrical = False
        self._total_extent = self._total_length
        self._current_density = float(current_density)
        self._temperature_ev = temperature_ev
        self._offset = float(offset)
        self._rng = as_generator(rng, backend="numpy")
        self._sampling = sampling
        self._mean_bias: float | None = None

    def _draw_points(self, num: int) -> np.ndarray:
        """Uniform emission points as a ``(3, num)`` array of ``x, y, segment`` rows."""
        rng = self._rng
        segment = np.searchsorted(self._cumulative, rng.random(num), side="right")
        segment = np.minimum(segment, self._starts.shape[0] - 1)
        along = rng.random(num)[:, None]
        starts = self._starts[segment]
        ends = self._ends[segment]
        if self._cylindrical:
            # points uniform over the swept area: density along the segment ~ r
            r1, r2 = starts[:, 1:2], ends[:, 1:2]
            radius = np.sqrt(r1**2 + along * (r2**2 - r1**2))
            dr = r2 - r1
            flat = np.abs(dr) < 1e-15 * np.maximum(np.abs(r1) + np.abs(r2), 1e-300)
            along = np.where(flat, along, (radius - r1) / np.where(flat, 1.0, dr))
        points = (
            starts + along * (ends - starts) + self._offset * self._normals[segment]
        )
        return np.vstack([points.T, segment])

    @property
    def sampling(self) -> ImportanceSampling | None:
        """Return the importance sampling of emitted markers, if any."""
        return self._sampling

    @property
    def cylindrical(self) -> bool:
        """Return whether the surface is one of revolution about the x axis (rings)."""
        return self._cylindrical

    @cylindrical.setter
    def cylindrical(self, value: bool) -> None:
        """Make the surface one of revolution about the x axis, or not."""
        self._cylindrical = bool(value)
        # a segment from r1 to r2 of length L sweeps the area pi (r1 + r2) L
        extent = (
            np.pi * (self._starts[:, 1] + self._ends[:, 1]) * self._lengths
            if value
            else self._lengths
        )
        self._total_extent = float(extent.sum())
        self._cumulative = np.cumsum(extent) / self._total_extent

    @property
    def total_length(self) -> float:
        """Return the emitting perimeter [m] (per metre of depth)."""
        return self._total_length

    @property
    def total_area(self) -> float:
        """Return the emitting area [m^2] of the surface of revolution (cylindrical)."""
        if not self._cylindrical:
            raise ValueError("total_area needs a cylindrical surface source")
        return self._total_extent

    @property
    def current_density(self) -> float:
        """Return the emitted current density [A/m^2]."""
        return self._current_density

    @property
    def rate(self) -> float:
        """Return the emitted particles per second (per metre of depth, or in total)."""
        extent = self._total_extent if self._cylindrical else self._total_length
        return self._current_density * extent / constants.e

    @property
    def segments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the selected segments: starts ``(s, 2)``, ends and outward normals."""
        return self._starts, self._ends, self._normals

    def markers_per_step(self, weight: float, dt: float) -> int:
        """Return the marker count of one step, carrying the fractional remainder.

        Parameters
        ----------
        weight : float
            Physical particles per marker.
        dt : float
            The time step [s].

        Returns
        -------
        int
            The number of markers.
        """
        return self._count(self.rate * self._flux_fraction * dt / weight)

    def sample(self, num: int, thermal_velocity: float) -> tuple[Array, Array]:
        """Return ``num`` positions ``(2, num)`` and velocities ``(2, num)``.

        Parameters
        ----------
        num : int
            The number of markers.
        thermal_velocity : float
            The emission's thermal speed [m/s].

        Returns
        -------
        tuple of numpy.ndarray
            The positions and the velocities.
        """
        positions, velocities, _ = self.sample_weighted(num, thermal_velocity)
        return positions, velocities

    def sample_weighted(
        self, num: int, thermal_velocity: float
    ) -> tuple[Array, Array, Array]:
        """Return positions, velocities and weight factors ``f / s`` of ``num`` markers.

        Parameters
        ----------
        num : int
            The number of markers.
        thermal_velocity : float
            The emission's thermal speed [m/s].

        Returns
        -------
        tuple of numpy.ndarray
            Positions ``(2, num)``, velocities ``(2, num)`` and factors ``(num,)``.
        """
        rng = self._rng
        sampling = self._sampling
        if sampling is not None and self._cylindrical:
            raise NotImplementedError(
                "importance sampling is not available in cylindrical coordinates"
            )
        if sampling is None:
            points = self._draw_points(num)
            factors = np.ones(num)
        else:
            # the bias sees the emission point, not the segment index
            bias = sampling.position_bias
            biased = ImportanceSampling(
                position_bias=None if bias is None else lambda rows: bias(rows[:2])
            )
            if self._mean_bias is None:
                self._mean_bias = biased.mean_bias(self._draw_points)
            points, factors = biased.sample_positions(
                rng, num, self._draw_points, self._mean_bias
            )
        positions = points[:2]
        normals = self._normals[points[2].astype(int)]
        if sampling is None:
            v_normal = flux_speeds(rng, num, thermal_velocity)
            v_tangent = rng.normal(0.0, thermal_velocity, num)
        else:
            v_normal, normal_factors = sampling.sample_flux_speeds(
                rng, num, thermal_velocity
            )
            v_tangent, tangent_factors = sampling.sample_transverse(
                rng, num, thermal_velocity
            )
            factors = factors * normal_factors * tangent_factors
        tangents = np.column_stack([-normals[:, 1], normals[:, 0]])
        velocities = v_normal[:, None] * normals + v_tangent[:, None] * tangents
        return positions, velocities.T, factors

    def emit(self, dt: float, species: MarkerSpecies) -> tuple[Array, Array, Array]:
        """Return one step's markers (see `_Source.emit`)."""
        if species.ndim != 2:
            raise ValueError("SurfaceSource only supports 2-D species")
        num = self.markers_per_step(species.weight, dt)
        if num == 0:
            return _empty(species)
        thermal_velocity = species.thermal_velocity_at(self._temperature_ev)
        positions, velocities, factors = self.sample_weighted(num, thermal_velocity)
        if species.nvel > 2:
            extra = np.zeros((species.nvel - 2, num))
            if self._cylindrical:
                # the azimuthal velocity, tangential to the surface of revolution
                extra[0] = self._rng.normal(0.0, thermal_velocity, num)
            velocities = np.concatenate([velocities, extra], axis=0)
        return (
            xp.asarray(positions),
            xp.asarray(velocities),
            xp.asarray(species.weight * factors),
        )


class SurfaceConversionSource(SurfaceSource):
    """Re-emit particles absorbed on a 2-D object as another species.

    Surface production of negative ions: positive ions (or atoms) that hit a caesiated
    plasma grid leave it as H- with a probability, the conversion yield. Every physical
    particle absorbed on ``obj`` (recorded with `add_impacts`) produces
    ``conversion_yield`` emitted particles. They leave from the point of the selected
    segments nearest to the impact, with the flux-weighted half-Maxwellian of
    `SurfaceSource` at ``temperature_ev``, at the next `emit`; fractional markers carry
    over.

    The emission follows the projectile flux, so it is not shared out over the ranks
    (``shares_flux``): each rank emits the products of its own impacts.

    Parameters
    ----------
    obj : object
        The 2-D object.
    projectile : str
        Name of the absorbed species (for the caller's bookkeeping).
    conversion_yield : float
        Emitted particles per absorbed one.
    target : int, optional
        The caller's loss target of ``obj``.
    temperature_ev : float, optional
        The emission temperature [eV].
    select : callable, optional
        ``select(midpoint, normal)`` is True for emitting segments.
    domain_bounds : tuple, optional
        ``(lower, upper)`` of the domain.
    offset : float
        Distance of the emission points outside the surface [m].
    rng : generator, int or None
        A NumPy generator, or a seed for a new one.
    """

    #: The emission is not a fixed rate the ranks share (see the class doc).
    shares_flux = False

    def __init__(
        self,
        obj: object,
        projectile: str,
        conversion_yield: float,
        *,
        target: int | None = None,
        temperature_ev: float | None = None,
        select: Callable[[Array, Array], bool] | None = None,
        domain_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
        offset: float = 1.0e-6,
        rng: Any = None,
    ) -> None:
        """Initialize the conversion source."""
        if conversion_yield < 0.0:
            raise ValueError("conversion_yield must be >= 0")
        super().__init__(
            obj,
            0.0,
            temperature_ev=temperature_ev,
            select=select,
            domain_bounds=domain_bounds,
            offset=offset,
            rng=rng,
        )
        self.projectile = str(projectile)
        self.target = target
        self.conversion_yield = float(conversion_yield)
        self._points: list[np.ndarray] = []
        self._segments: list[np.ndarray] = []
        self._weights: list[np.ndarray] = []
        self._pending = 0.0  # physical particles produced but not yet emitted
        #: physical particles absorbed and emitted since the start (this rank), and the
        #: emitted ones per selected segment (``segments``)
        self.absorbed = 0.0
        self.emitted = 0.0
        self.emitted_per_segment = np.zeros(self._starts.shape[0])

    @property
    def rate(self) -> float:
        """Return 0: the emission follows the absorbed projectiles, not a rate."""
        return 0.0

    def nearest_surface(self, positions: Array) -> tuple[np.ndarray, np.ndarray]:
        """Return the nearest point ``(n, 2)`` on the selected segments, and its segment.

        Parameters
        ----------
        positions : array, shape (2, n)
            The positions.

        Returns
        -------
        tuple of numpy.ndarray
            The nearest points ``(n, 2)`` and their segment indices.
        """
        points = np.asarray(xp.to_numpy(positions), dtype=float).T  # (n, 2)
        starts, ends = self._starts, self._ends
        edge = ends - starts  # (s, 2)
        relative = points[:, None, :] - starts[None, :, :]  # (n, s, 2)
        length2 = np.maximum(np.sum(edge**2, axis=1), 1e-300)
        along = np.clip(np.sum(relative * edge[None], axis=2) / length2, 0.0, 1.0)
        nearest = starts[None] + along[..., None] * edge[None]  # (n, s, 2)
        distance = np.sum((points[:, None, :] - nearest) ** 2, axis=2)
        segment = np.argmin(distance, axis=1)
        return nearest[np.arange(points.shape[0]), segment], segment

    def add_impacts(self, positions: Array, weights: Array) -> None:
        """Record projectiles absorbed on the object.

        Parameters
        ----------
        positions : array, shape (2, n)
            Where they were absorbed.
        weights : array, shape (n,)
            Their weights.
        """
        weights = np.asarray(xp.to_numpy(weights), dtype=float)
        if weights.size == 0:
            return
        points, segment = self.nearest_surface(positions)
        self._points.append(points)
        self._segments.append(segment)
        self._weights.append(weights)
        self.absorbed += float(weights.sum())

    def emit(self, dt: float, species: MarkerSpecies) -> tuple[Array, Array, Array]:
        """Return the products of the recorded impacts; ``dt`` is unused."""
        if species.ndim != 2:
            raise ValueError("SurfaceConversionSource only supports 2-D species")
        if self._weights:
            points = np.concatenate(self._points)
            segments = np.concatenate(self._segments)
            impact_weights = np.concatenate(self._weights)
            self._points, self._segments, self._weights = [], [], []
        else:
            points = np.zeros((0, 2))
            segments = np.zeros(0, dtype=np.int64)
            impact_weights = np.zeros(0)
        produced = self._pending + self.conversion_yield * float(impact_weights.sum())
        weight = float(species.weight)
        num = int(np.floor(produced / weight)) if impact_weights.size else 0
        self._pending = produced - num * weight
        if num == 0:
            return _empty(species)
        rng = self._rng
        # each marker comes from an impact drawn by its weight
        chosen = rng.choice(
            impact_weights.size, size=num, p=impact_weights / impact_weights.sum()
        )
        normals = self._normals[segments[chosen]]
        positions = points[chosen] + self._offset * normals
        thermal_velocity = species.thermal_velocity_at(self._temperature_ev)
        v_normal = flux_speeds(rng, num, thermal_velocity)
        v_tangent = rng.normal(0.0, thermal_velocity, num)
        tangents = np.column_stack([-normals[:, 1], normals[:, 0]])
        velocities = (v_normal[:, None] * normals + v_tangent[:, None] * tangents).T
        if species.nvel > 2:
            extra = rng.normal(0.0, thermal_velocity, (species.nvel - 2, num))
            velocities = np.concatenate([velocities, extra], axis=0)
        self.emitted += num * weight
        self.emitted_per_segment += weight * np.bincount(
            segments[chosen], minlength=self._starts.shape[0]
        )
        return xp.asarray(positions.T), xp.asarray(velocities), xp.full(num, weight)


def _unit(vector: Sequence[float], name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise ValueError(f"{name} must have a nonzero length")
    return value / norm


def orthonormal_frame(
    axis: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors normal to ``axis`` and to each other.

    Parameters
    ----------
    axis : sequence of float
        A unit 3-vector.

    Returns
    -------
    tuple of numpy.ndarray
        ``(first, second)`` with ``(first, second, axis)`` right-handed.
    """
    axis_array = np.asarray(axis, dtype=float)
    helper = np.eye(3)[int(np.argmin(np.abs(axis_array)))]
    first = np.cross(axis_array, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(axis_array, first)


@method(
    equations={
        "intensity": r"\frac{dN}{d\Omega} = \frac{n + 1}{2\pi}\cos^n\theta, \quad 0 \le \theta \le \frac{\pi}{2}, \qquad \int_{2\pi} \frac{dN}{d\Omega}\, d\Omega = 1",
        "direction": r"\cos\theta = u^{1/(n + 1)}, \quad \varphi = 2\pi u', \qquad u, u' \sim U(0, 1)",
        "speeds": r"f(v) \propto v^3 e^{-v^2/2v_\mathrm{th}^2}: \quad v = v_\mathrm{th}\sqrt{-2\ln(u_1 u_2)}, \qquad v_\mathrm{th} = \sqrt{e T_\mathrm{eV}/m}",
    },
    assumptions=(
        r"$n = 1$ the cosine law of an orifice, $n = 2$ the profile of the CsFlow3D caesium nozzle; or a table of $\theta$ and intensity, linear in between",
        r"the speeds are those of the flux of a Maxwellian through a surface, drawn independently of the direction",
        r"emitted from a point or a uniform disc normal to the axis",
    ),
    references=(
        "[Gutser, Wünderlich and Fantz, Plasma Phys. Control. Fusion 53, 105014 (2011)](https://doi.org/10.1088/0741-3335/53/10/105014)",
    ),
)
class DirectedSource(_CountedSource):
    """Emit markers from ``centre`` (a disc of ``radius`` normal to ``axis``) about ``axis``.

    The direction makes the angle ``theta`` with the axis and follows an intensity per
    solid angle ``cos(theta)**exponent`` (1 is the cosine law of an orifice, 2 the cos^2
    profile of the CsFlow3D caesium nozzle, Gutser et al. 2011) or a table of ``theta``
    and intensity. The speeds are those of the flux of a Maxwellian through a surface
    at ``temperature_ev``, drawn independently of the direction. `sample` returns
    markers to place at once (a burst, each marker carrying ``rate / N`` per second,
    `marker_weight`); `emit` emits ``rate`` physical particles per second, step by step,
    into a 3D3V species. The draws are on the host (a NumPy generator).

    Parameters
    ----------
    centre : sequence of float
        Centre of the emitting point or disc [m], a 3-vector.
    axis : sequence of float
        Direction of the emission axis (normalised here); nothing is emitted backwards.
    rate : float
        Physical particles emitted per second.
    temperature_ev : float
        Temperature of the Maxwellian whose flux sets the speeds [eV].
    exponent : float
        ``n`` of the intensity ``cos(theta)**n``, ``n > -1``.
    angular_table : tuple of sequence, optional
        ``(theta, intensity)``: the intensity per solid angle at increasing angles in
        radians from 0 to at most pi / 2, linear in between and zero beyond the last
        angle. Replaces ``exponent``.
    radius : float
        Radius of the emitting disc [m]; 0 is a point.
    rng : generator, int or None
        A NumPy generator, or a seed for a new one.
    """

    @staticmethod
    def _cdf(
        theta: Sequence[float], intensity: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fine grid of ``theta`` and the cumulative emission from the axis."""
        angles = np.asarray(theta, dtype=float)
        values = np.asarray(intensity, dtype=float)
        if angles.ndim != 1 or angles.shape != values.shape or angles.size < 2:
            raise ValueError("angular_table needs two 1D arrays of equal length >= 2")
        if np.any(np.diff(angles) <= 0) or angles[0] < 0 or angles[-1] > np.pi / 2:
            raise ValueError("table angles must increase within [0, pi / 2]")
        if np.any(values < 0) or not np.any(values > 0):
            raise ValueError("table intensities must be >= 0, some positive")
        fine = np.linspace(angles[0], angles[-1], 4097)
        weight = np.interp(fine, angles, values) * np.sin(fine)
        cumulative = np.concatenate(
            [[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(fine))]
        )
        # intensity per particle: the table over its integral over the solid angle
        norm = 2 * np.pi * cumulative[-1]
        return fine, cumulative / cumulative[-1], angles, values / norm

    def __init__(
        self,
        centre: Sequence[float],
        axis: Sequence[float],
        *,
        rate: float,
        temperature_ev: float,
        exponent: float = 2.0,
        angular_table: tuple[Sequence[float], Sequence[float]] | None = None,
        radius: float = 0.0,
        rng: Any = None,
    ) -> None:
        """Set up the source."""
        if not rate >= 0.0:
            raise ValueError("rate must be >= 0")
        if not temperature_ev > 0.0:
            raise ValueError("temperature_ev must be positive")
        if not exponent > -1.0:
            raise ValueError("exponent must be > -1")
        if not radius >= 0.0:
            raise ValueError("radius must be >= 0")
        self._centre = np.asarray(centre, dtype=float)
        if self._centre.shape != (3,):
            raise ValueError("centre must be a 3-vector")
        self._axis = _unit(axis, "axis")
        self._first, self._second = orthonormal_frame(self._axis)
        self._rate = float(rate)
        self._temperature_ev = float(temperature_ev)
        self._exponent = float(exponent)
        self._radius = float(radius)
        self._table = None if angular_table is None else self._cdf(*angular_table)
        self._rng = as_generator(rng, backend="numpy")

    def _cos_theta(self, num: int) -> np.ndarray:
        u = self._rng.random(size=num)
        if self._table is None:
            return u ** (1.0 / (self._exponent + 1.0))
        fine, cumulative, _, _ = self._table
        return np.cos(np.interp(u, cumulative, fine))

    @property
    def centre(self) -> tuple[float, ...]:
        """Return the centre of the emitter."""
        return tuple(float(c) for c in self._centre)

    @property
    def axis(self) -> tuple[float, ...]:
        """Return the unit emission axis."""
        return tuple(float(a) for a in self._axis)

    @property
    def temperature_ev(self) -> float:
        """Return the temperature of the Maxwellian whose flux sets the speeds [eV]."""
        return self._temperature_ev

    @property
    def rate(self) -> float:
        """Return the emitted physical particles per second."""
        return self._rate

    def intensity(self, cos_theta: Array) -> Array:
        """Return ``dN/dOmega`` per emitted particle at ``cos(theta)`` [1/sr].

        Integrates to 1 over the forward half-space, so ``rate * intensity / r**2`` is
        the flux density of a point source at the distance ``r`` along that direction.

        Parameters
        ----------
        cos_theta : array
            Cosines of the angle to the axis.

        Returns
        -------
        numpy.ndarray
            The intensity.
        """
        mu = np.asarray(cos_theta, dtype=float)
        forward = mu > 0
        if self._table is None:
            n = self._exponent
            value = (n + 1) / (2 * np.pi) * np.where(forward, mu, 0.0) ** n
            return np.where(forward, value, 0.0)
        _, _, angles, values = self._table
        theta = np.arccos(np.clip(mu, -1.0, 1.0))
        inside = forward & (theta <= angles[-1]) & (theta >= angles[0])
        return np.where(inside, np.interp(theta, angles, values), 0.0)

    def speeds(self, num: int, mass: float) -> np.ndarray:
        """Return ``num`` speeds of the flux of a Maxwellian (``v**3`` times the Gaussian).

        Parameters
        ----------
        num : int
            The number of speeds.
        mass : float
            Mass of a particle [kg].

        Returns
        -------
        numpy.ndarray
            The speeds [m/s].
        """
        v_th = np.sqrt(constants.e * self._temperature_ev / mass)
        # v**2 / (2 v_th**2) follows a Gamma(2) distribution: a sum of two exponentials
        u = 1.0 - self._rng.random(size=(2, num))
        return v_th * np.sqrt(-2.0 * np.log(u[0] * u[1]))

    def sample(self, num: int, mass: float) -> tuple[Array, Array]:
        """Return ``(positions, velocities)`` of ``num`` markers, ``(3, num)`` each.

        Parameters
        ----------
        num : int
            The number of markers.
        mass : float
            Mass of a particle [kg].

        Returns
        -------
        tuple of array
            The positions and the velocities, on the active backend.
        """
        mu = self._cos_theta(num)
        phi = 2 * np.pi * self._rng.random(size=num)
        sine = np.sqrt(np.clip(1.0 - mu * mu, 0.0, None))
        direction = (
            mu[None, :] * self._axis[:, None]
            + (sine * np.cos(phi))[None, :] * self._first[:, None]
            + (sine * np.sin(phi))[None, :] * self._second[:, None]
        )
        velocities = direction * self.speeds(num, mass)[None, :]
        positions = np.repeat(self._centre[:, None], num, axis=1)
        if self._radius > 0:
            r = self._radius * np.sqrt(self._rng.random(size=num))
            angle = 2 * np.pi * self._rng.random(size=num)
            positions = (
                positions
                + (r * np.cos(angle))[None, :] * self._first[:, None]
                + (r * np.sin(angle))[None, :] * self._second[:, None]
            )
        return xp.asarray(positions), xp.asarray(velocities)

    def marker_weight(self, num: int) -> float:
        """Return the weight (particles per second) of each of ``num`` burst markers.

        Parameters
        ----------
        num : int
            The number of burst markers.

        Returns
        -------
        float
            ``rate / num``.
        """
        return self._rate / num

    def emit(self, dt: float, species: MarkerSpecies) -> tuple[Array, Array, Array]:
        """Return ``rate * dt`` physical particles of a 3D3V species as markers."""
        if species.ndim != 3 or species.nvel != 3:
            raise ValueError("DirectedSource needs a 3D species with nvel = 3")
        num = self._count(self._rate * self._flux_fraction * dt / species.weight)
        if num == 0:
            return _empty(species)
        positions, velocities = self.sample(num, species.mass)
        return positions, velocities, xp.full(num, species.weight)


__all__ = [
    "ChargeBalanceSource",
    "DirectedSource",
    "MarkerSpecies",
    "PlanarFluxInjector1D",
    "PlanarFluxInjector2D",
    "ProfiledVolumeSource",
    "ReemissionSource",
    "SurfaceConversionSource",
    "SurfaceSource",
    "UniformVolumeSource",
    "cosine_profile",
    "object_surface_segments",
    "orthonormal_frame",
    "planar_flux_injector",
]
