"""Draw markers: Maxwellian and flux-weighted velocities, quiet starts, importance sampling.

Every draw takes a random generator, ``rng``: a ``numpy.random.Generator`` or a
``cupy.random.Generator`` (CuPy's lacks some of NumPy's methods, so `normal` and
`uniform` derive them from ``random`` and ``standard_normal``). A code that seeds its
generators keeps its runs reproducible; `make_generator` creates one from a seed on the
active cunumpy backend.

`maxwellian_velocities` draws a Maxwellian, `flux_speeds` the normal speeds of the
one-way flux of a Maxwellian through a plane, and `quiet_start` a low-discrepancy set
of positions and velocities.

Without importance sampling, markers are drawn from the physical distribution ``f``
itself and every marker of a species carries the same weight
``W0 = N_phys / N_markers``, so a sparse part of phase space (the tail of a Maxwellian,
a low-density region) gets few markers. `ImportanceSampling` draws them from a sampling
distribution ``s`` instead and gives each one the weight factor ``f / s``, whose mean
under ``s`` is one: the expected total weight is still ``N_phys`` and every weighted
moment is unbiased. Where ``s`` is larger than ``f`` there are more, lighter markers.

- ``temperature_ratio``: velocities drawn at ``T_s = temperature_ratio * T``; above one
  puts more markers into the tail, and for a Maxwellian the factors are bounded by
  ``temperature_ratio ** (d / 2)``.
- ``drift``: the sampling Maxwellian is shifted by this velocity (volume loading only).
- ``position_bias``: a relative marker density ``g(x)`` in ``(0, 1]``. Candidate
  positions from ``f`` are kept with probability ``g`` and the factor is
  ``<g> / g(x)``, where ``<g>`` is the mean of ``g`` over the region. `PositionBias`
  is a piecewise-linear ``g`` along one axis, with an exact ``<g>`` over a box.

The weights are unbiased, so the total weight of a batch fluctuates by about
``std(f / s) / sqrt(N)``; an initial loading can be rescaled to the exact total
(``normalize_loading``, applied by the caller).

Examples
--------
>>> import numpy as np
>>> from mpiparticles.sampling import ImportanceSampling, maxwellian_velocities
>>> rng = np.random.default_rng(1)
>>> maxwellian_velocities(rng, 4, 3, 1.0).shape
(3, 4)
>>> velocities, factors = ImportanceSampling(temperature_ratio=2.0).sample_maxwellian(
...     rng, 1000, 3, 1.0
... )
>>> bool(abs(factors.mean() - 1.0) < 0.1)
True
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cunumpy as xp
import numpy as np
from methodcard import method
from scipy.special import ndtri

from mpiparticles._types import Array

#: Number of candidate positions used to estimate ``<g>`` when it has no closed form
#: (a general callable, or a non-box region).
MEAN_BIAS_SAMPLES = 1 << 16

#: The bases of the radical inverses of `quiet_start`.
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


def make_generator(seed: int | None = None, backend: str | None = None) -> Any:
    """Return a new random generator of ``backend`` (the active cunumpy one by default).

    Parameters
    ----------
    seed : int, optional
        The seed; None seeds from the operating system.
    backend : str, optional
        ``"numpy"`` for a host generator whatever the active backend; the active backend of
        cunumpy (CuPy's generator on the GPU) by default.

    Returns
    -------
    numpy.random.Generator or cupy.random.Generator
        The generator.
    """
    backend = xp.get_backend() if backend is None else backend
    if backend == "numpy":
        return np.random.default_rng(seed)
    return xp.random.default_rng(seed)


def as_generator(rng: Any, backend: str | None = None) -> Any:
    """Return ``rng`` if it is a generator, else a new one seeded with it.

    Parameters
    ----------
    rng : generator, int or None
        A NumPy or CuPy generator, used as it is; an int seeds a new one; None gives an
        unseeded one.
    backend : str, optional
        Backend of a new generator (`make_generator`).

    Returns
    -------
    numpy.random.Generator or cupy.random.Generator
        The generator.
    """
    if rng is None or isinstance(rng, (int, np.integer)):
        return make_generator(None if rng is None else int(rng), backend)
    return rng


def normal(
    rng: Any,
    loc: float | Array = 0.0,
    scale: float | Array = 1.0,
    size: int | tuple[int, ...] | None = None,
) -> Array:
    """Return normal samples from ``rng``, also from a CuPy generator.

    Parameters
    ----------
    rng : numpy.random.Generator or cupy.random.Generator
        The generator.
    loc, scale : float or array
        Mean and standard deviation.
    size : int or tuple of int, optional
        Shape of the samples.

    Returns
    -------
    array
        The samples, on the generator's device.
    """
    if hasattr(rng, "normal"):
        return rng.normal(loc=loc, scale=scale, size=size)
    return loc + scale * rng.standard_normal(size=size)  # pragma: no cover - CuPy


def uniform(
    rng: Any,
    low: float | Array = 0.0,
    high: float | Array = 1.0,
    size: int | tuple[int, ...] | None = None,
) -> Array:
    """Return uniform samples in ``[low, high)`` from ``rng``, also from a CuPy generator.

    Parameters
    ----------
    rng : numpy.random.Generator or cupy.random.Generator
        The generator.
    low, high : float or array
        The bounds.
    size : int or tuple of int, optional
        Shape of the samples.

    Returns
    -------
    array
        The samples, on the generator's device.
    """
    if hasattr(rng, "uniform"):
        return rng.uniform(low=low, high=high, size=size)
    return low + (high - low) * rng.random(size=size)  # pragma: no cover - CuPy


def _namespace(array: Array) -> Any:
    """Return NumPy for host arrays and the cunumpy backend otherwise."""
    return np if isinstance(array, np.ndarray) else xp


def _host_float(value: Any) -> float:
    """Return a scalar, also a device one, as a Python float."""
    return float(xp.to_numpy(value) if xp.is_gpu(value) else value)


def maxwellian_velocities(
    rng: Any, num_markers: int, nvel: int, thermal_velocity: float
) -> Array:
    """Return ``(nvel, num_markers)`` velocities of a zero-mean Maxwellian.

    Parameters
    ----------
    rng : numpy.random.Generator or cupy.random.Generator
        The generator.
    num_markers : int
        The number of markers.
    nvel : int
        The velocity components.
    thermal_velocity : float
        ``v_th = sqrt(T / m)``, the standard deviation of each component.

    Returns
    -------
    array, shape (nvel, num_markers)
        The velocities.
    """
    return normal(rng, 0.0, float(thermal_velocity), (nvel, num_markers))


@method(
    equations={
        "flux": r"f(v) = \frac{v}{v_\mathrm{th}^2}\, e^{-v^2/2v_\mathrm{th}^2}, \quad v > 0",
        "inverse_cdf": r"v = v_\mathrm{th}\sqrt{-2\ln(1 - u)}, \quad u \sim U(0, 1)",
    },
    assumptions=(
        r"the normal speeds of the one-way flux of a Maxwellian through a plane (a Rayleigh distribution)",
    ),
)
def flux_speeds(rng: Any, num_markers: int, thermal_velocity: float) -> Array:
    """Return normal speeds from the flux-weighted half-Maxwellian.

    Parameters
    ----------
    rng : numpy.random.Generator or cupy.random.Generator
        The generator.
    num_markers : int
        The number of markers.
    thermal_velocity : float
        ``v_th = sqrt(T / m)``.

    Returns
    -------
    array, shape (num_markers,)
        Speeds ``v >= 0`` with the density ``v / v_th^2 exp(-v^2 / 2 v_th^2)``.
    """
    aux = rng.random(num_markers)
    mod = _namespace(aux)
    return float(thermal_velocity) * mod.sqrt(
        -2.0 * mod.log(1.0 - mod.clip(aux, 0.0, 1.0 - 1e-15))
    )


class PositionBias:
    """Piecewise-linear relative marker density along one axis.

    ``g`` is the linear interpolant of ``values`` at ``points`` along ``axis``, held
    constant beyond the first and last point. Values must lie in ``(0, 1]``: ``g = 1``
    keeps every candidate marker, ``g = 0.1`` keeps one in ten (and gives those ten
    times the weight).

    Parameters
    ----------
    points : sequence of float
        Strictly increasing interpolation points.
    values : sequence of float
        The relative marker densities at the points, in ``(0, 1]``.
    axis : int
        The position axis the bias varies along.
    """

    def __init__(
        self,
        points: Sequence[float],
        values: Sequence[float],
        axis: int = 0,
    ) -> None:
        """Initialize the position bias."""
        points_array = np.asarray(points, dtype=float)
        values_array = np.asarray(values, dtype=float)
        if (
            points_array.ndim != 1
            or points_array.shape != values_array.shape
            or points_array.size == 0
        ):
            raise ValueError("points and values must be 1-D and of equal length")
        if np.any(np.diff(points_array) <= 0.0):
            raise ValueError("points must be strictly increasing")
        if np.any(values_array <= 0.0) or np.any(values_array > 1.0):
            raise ValueError("position-bias values must lie in (0, 1]")
        if axis < 0:
            raise ValueError("axis must be >= 0")
        self._points = points_array
        self._values = values_array
        self._axis = int(axis)

    @property
    def points(self) -> np.ndarray:
        """Return the interpolation points."""
        return self._points

    @property
    def values(self) -> np.ndarray:
        """Return the relative marker densities at the points."""
        return self._values

    @property
    def axis(self) -> int:
        """Return the axis the bias varies along."""
        return self._axis

    def mean_over_box(
        self, lower_bound: Sequence[float], upper_bound: Sequence[float]
    ) -> float:
        """Return the exact mean of ``g`` over a box (uniform positions).

        Parameters
        ----------
        lower_bound, upper_bound : sequence of float
            The corners of the box.

        Returns
        -------
        float
            The mean ``<g>``.
        """
        lo = float(lower_bound[self._axis])
        hi = float(upper_bound[self._axis])
        if hi <= lo:
            return float(np.interp(lo, self._points, self._values))
        inner = self._points[(self._points > lo) & (self._points < hi)]
        nodes = np.concatenate(([lo], inner, [hi]))
        g = np.interp(nodes, self._points, self._values)
        return float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(nodes)) / (hi - lo))

    def __call__(self, positions: Array) -> Array:
        """Return ``g`` at ``positions`` of shape ``(ndim, N)``."""
        mod = _namespace(positions)
        return mod.interp(
            positions[self._axis],
            mod.asarray(self._points),
            mod.asarray(self._values),
        )


@dataclass(frozen=True)
@method(
    equations={
        "weight": r"w = W_0\,\frac{f(\mathbf{x}, \mathbf{v})}{s(\mathbf{x}, \mathbf{v})}, \qquad W_0 = \frac{N_\mathrm{phys}}{N_\mathrm{markers}}, \qquad \langle f/s\rangle_s = 1",
        "temperature": r"\frac{f}{s} = \left(\frac{v_s}{v_\mathrm{th}}\right)^{d} \exp\!\left[-\frac{|\mathbf{v}|^2}{2v_\mathrm{th}^2} + \frac{|\mathbf{v} - \mathbf{u}_s|^2}{2v_s^2}\right], \qquad v_s = \sqrt{T_\mathrm{ratio}}\,v_\mathrm{th}",
        "flux": r"\frac{f}{s} = T_\mathrm{ratio}\, \exp\!\left[\frac{v^2}{2}\left(\frac{1}{v_s^2} - \frac{1}{v_\mathrm{th}^2}\right)\right] \quad \text{(flux speeds)}",
        "position": r"\frac{f}{s} = \frac{\langle g\rangle}{g(\mathbf{x})}, \qquad 0 < g \le 1",
    },
    assumptions=(
        r"$f$ a zero-mean Maxwellian of $v_\mathrm{th} = \sqrt{T/m}$ per component in $d$ velocity dimensions; $s$ the same at $T_\mathrm{ratio} T$, shifted by $\mathbf{u}_s$ (volume loading and sources only)",
        r"positions from $f$ kept with probability $g(\mathbf{x})$; $\langle g\rangle$ exact over a box for the piecewise-linear `PositionBias`",
        r"the weights are unbiased; an initial loading may be rescaled to the exact total weight (`normalize_loading`), sources are not",
    ),
)
class ImportanceSampling:
    """Sampling distribution ``s`` used instead of the physical ``f``.

    Parameters
    ----------
    temperature_ratio : float
        Sampling temperature over physical temperature (``> 0``).
    drift : tuple of float, optional
        Velocity shift of the sampling Maxwellian, one entry per velocity component.
        Only volume loading and volume sources use it.
    position_bias : callable, optional
        Relative marker density ``g(x)`` in ``(0, 1]``, called with positions of shape
        ``(ndim, N)``; for example a `PositionBias`.
    normalize_loading : bool
        Whether an initial loading rescales its weights so that their sum is exactly
        ``N_markers * W0`` (a bias of order ``1 / N``). Sources ignore it.
    """

    temperature_ratio: float = 1.0
    drift: tuple[float, ...] | None = None
    position_bias: Callable[[Array], Array] | None = None
    normalize_loading: bool = True

    def __post_init__(self) -> None:
        """Validate the sampling parameters."""
        if not self.temperature_ratio > 0.0:
            raise ValueError("temperature_ratio must be > 0")
        if self.drift is not None:
            object.__setattr__(self, "drift", tuple(float(v) for v in self.drift))

    def _drift_vector(self, nvel: int) -> np.ndarray:
        if self.drift is None:
            return np.zeros(nvel)
        if len(self.drift) != nvel:
            raise ValueError(
                f"drift has {len(self.drift)} components, the species has {nvel}"
            )
        return np.asarray(self.drift, dtype=float)

    def _evaluate_bias(self, positions: Array) -> Array:
        assert self.position_bias is not None
        g = self.position_bias(positions)
        mod = _namespace(g)
        if bool(mod.any(g <= 0.0)) or bool(mod.any(g > 1.0 + 1e-12)):
            raise ValueError("position_bias must return values in (0, 1]")
        return g

    def _reject_drift(self, where: str) -> None:
        if self.has_drift:
            raise ValueError(f"a sampling drift is not available for {where}")

    @property
    def has_drift(self) -> bool:
        """Return whether the sampling Maxwellian is shifted."""
        return self.drift is not None and any(value != 0.0 for value in self.drift)

    @property
    def biases_velocities(self) -> bool:
        """Return whether velocities are drawn from something other than ``f``."""
        return self.temperature_ratio != 1.0 or self.has_drift

    @property
    def biases_positions(self) -> bool:
        """Return whether positions are drawn from something other than ``f``."""
        return self.position_bias is not None

    @property
    def is_trivial(self) -> bool:
        """Return whether ``s == f`` (every weight factor is one)."""
        return not (self.biases_velocities or self.biases_positions)

    def sample_maxwellian(
        self,
        rng: Any,
        num_markers: int,
        nvel: int,
        thermal_velocity: float,
    ) -> tuple[Array, Array]:
        """Draw velocities for a zero-mean Maxwellian ``f``.

        Parameters
        ----------
        rng : numpy.random.Generator or cupy.random.Generator
            The generator.
        num_markers : int
            The number of markers.
        nvel : int
            The velocity components.
        thermal_velocity : float
            ``v_th`` of ``f``.

        Returns
        -------
        velocities : array, shape (nvel, num_markers)
            The velocities.
        factors : array, shape (num_markers,)
            The weight factors ``f / s``.
        """
        drift = self._drift_vector(nvel)
        thermal_velocity = float(thermal_velocity)
        if thermal_velocity == 0.0 or not self.biases_velocities:
            velocities = maxwellian_velocities(rng, num_markers, nvel, thermal_velocity)
            return velocities, _namespace(velocities).ones(num_markers)
        sampling_velocity = thermal_velocity * np.sqrt(self.temperature_ratio)
        velocities = maxwellian_velocities(rng, num_markers, nvel, sampling_velocity)
        mod = _namespace(velocities)
        shift = mod.asarray(drift)[:, None]
        velocities = velocities + shift
        exponent = -0.5 * mod.sum(velocities**2, axis=0) / thermal_velocity**2
        exponent += (
            0.5 * mod.sum((velocities - shift) ** 2, axis=0) / sampling_velocity**2
        )
        prefactor = (sampling_velocity / thermal_velocity) ** nvel
        return velocities, prefactor * mod.exp(exponent)

    def sample_transverse(
        self,
        rng: Any,
        num_markers: int,
        thermal_velocity: float,
    ) -> tuple[Array, Array]:
        """Draw one Maxwellian velocity component (tangential to a flux source).

        Parameters
        ----------
        rng : numpy.random.Generator or cupy.random.Generator
            The generator.
        num_markers : int
            The number of markers.
        thermal_velocity : float
            ``v_th`` of ``f``.

        Returns
        -------
        velocities : array, shape (num_markers,)
            The component.
        factors : array, shape (num_markers,)
            The weight factors ``f / s``.
        """
        self._reject_drift("flux sources")
        velocities, factors = self.sample_maxwellian(
            rng, num_markers, 1, thermal_velocity
        )
        return velocities[0], factors

    def sample_flux_speeds(
        self,
        rng: Any,
        num_markers: int,
        thermal_velocity: float,
    ) -> tuple[Array, Array]:
        """Draw normal speeds from the flux-weighted half-Maxwellian (`flux_speeds`).

        The sampling distribution is the same at ``temperature_ratio * T``.

        Parameters
        ----------
        rng : numpy.random.Generator or cupy.random.Generator
            The generator.
        num_markers : int
            The number of markers.
        thermal_velocity : float
            ``v_th`` of ``f``.

        Returns
        -------
        speeds : array, shape (num_markers,)
            The speeds.
        factors : array, shape (num_markers,)
            The weight factors ``f / s``.
        """
        self._reject_drift("flux sources")
        thermal_velocity = float(thermal_velocity)
        sampling_velocity = thermal_velocity * np.sqrt(self.temperature_ratio)
        speeds = flux_speeds(rng, num_markers, sampling_velocity)
        mod = _namespace(speeds)
        if thermal_velocity == 0.0 or self.temperature_ratio == 1.0:
            return speeds, mod.ones(num_markers)
        exponent = (
            0.5 * speeds**2 * (1.0 / sampling_velocity**2 - 1.0 / thermal_velocity**2)
        )
        return speeds, self.temperature_ratio * mod.exp(exponent)

    def mean_bias(
        self,
        draw: Callable[[int], Array],
        lower_bound: Sequence[float] | None = None,
        upper_bound: Sequence[float] | None = None,
        num_samples: int = MEAN_BIAS_SAMPLES,
    ) -> float:
        """Return ``<g>``, the mean of the position bias under ``f``.

        Parameters
        ----------
        draw : callable
            ``draw(m)`` returns ``m`` candidate positions from ``f``, ``(ndim, m)``.
        lower_bound, upper_bound : sequence of float, optional
            The box ``f`` fills uniformly; with a `PositionBias` the mean is exact.
        num_samples : int
            Candidates of the estimate otherwise.

        Returns
        -------
        float
            ``<g>``; 1 without a position bias.
        """
        if self.position_bias is None:
            return 1.0
        mean_over_box = getattr(self.position_bias, "mean_over_box", None)
        if lower_bound is not None and upper_bound is not None and mean_over_box:
            return float(mean_over_box(lower_bound, upper_bound))
        g = self._evaluate_bias(draw(num_samples))
        return _host_float(_namespace(g).mean(g))

    def sample_positions(
        self,
        rng: Any,
        num_markers: int,
        draw: Callable[[int], Array],
        mean_bias: float,
    ) -> tuple[Array, Array]:
        """Draw positions from ``s ∝ g f`` by rejection.

        Parameters
        ----------
        rng : numpy.random.Generator or cupy.random.Generator
            The generator of the acceptance draws.
        num_markers : int
            The number of markers.
        draw : callable
            ``draw(m)`` returns ``m`` candidate positions from ``f``, ``(ndim, m)``.
        mean_bias : float
            ``<g>`` (`mean_bias`).

        Returns
        -------
        positions : array, shape (ndim, num_markers)
            The accepted positions.
        factors : array, shape (num_markers,)
            The weight factors ``<g> / g``.
        """
        if self.position_bias is None or num_markers == 0:
            positions = draw(num_markers)
            return positions, _namespace(positions).ones(num_markers)
        kept_positions = []
        kept_bias = []
        num_kept = 0
        while num_kept < num_markers:
            missing = num_markers - num_kept
            num_candidates = max(16, int(1.25 * missing / mean_bias) + 1)
            candidates = draw(num_candidates)
            g = self._evaluate_bias(candidates)
            keep = rng.random(num_candidates) < g
            kept_positions.append(candidates[:, keep])
            kept_bias.append(g[keep])
            num_kept += int(_host_float(keep.sum()))
        mod = _namespace(kept_positions[0])
        positions = mod.concatenate(kept_positions, axis=1)[:, :num_markers]
        g = mod.concatenate(kept_bias)[:num_markers]
        return positions, mean_bias / g


def radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    """Return the radical inverse (van der Corput sequence) of ``indices`` in ``base``.

    The digits of ``i`` in ``base`` mirrored about the point: ``i = sum d_j b^j`` gives
    ``sum d_j b^-(j+1)`` in ``[0, 1)``. For ``base = 2`` and ``N = 2^m`` points it is the
    bit-reversed permutation of ``i / N``.

    Parameters
    ----------
    indices : array of int
        The indices.
    base : int
        The base.

    Returns
    -------
    numpy.ndarray
        The radical inverses, of the shape of ``indices``.
    """
    digits = np.asarray(indices, dtype=np.int64).copy()
    result = np.zeros(digits.shape)
    scale = 1.0 / base
    while np.any(digits > 0):
        result += (digits % base) * scale
        digits //= base
        scale /= base
    return result


@method(
    equations={
        "points": r"u_{i,0} = \frac{i + 1/2}{N}, \qquad u_{i,j} = \phi_{p_j}(i) + \frac{1}{2 p_j^{K_j}} \quad (j \ge 1,\; p_j = 2, 3, 5, \ldots,\; K_j = \lceil \log_{p_j} N \rceil)",
        "load": r"\mathbf{x}_i = \mathbf{x}_\mathrm{lo} + (u_{i,0}, \ldots, u_{i,d-1}) \odot \mathbf{L}, \qquad v_{i,c} = \Phi^{-1}(u_{i,d+c})",
    },
    assumptions=(
        r"a Hammersley set in position and velocity: the first coordinate evenly spaced, the others radical inverses $\phi_p$ in distinct primes, each moved to the middle of its elementary interval so that no coordinate is 0 or 1",
        r"in 2D with $N = 2^m$ markers and $2^a \times 2^b$ cells ($a + b \le m$), every cell holds exactly $N / 2^{a+b}$ markers (a (0, m, 2)-net in base 2)",
        r"the velocities by the inverse normal CDF $\Phi^{-1}$; other primes for the velocities (``velocity_bases``) decorrelate species loaded on the same positions",
    ),
    references=(
        "[Hammersley and Handscomb, Monte Carlo Methods (1964)](https://doi.org/10.1007/978-94-009-5819-7)",
        "[Birdsall and Langdon, Plasma Physics via Computer Simulation, sec. 16-6 (1991)](https://doi.org/10.1201/9781315275048)",
    ),
)
def quiet_start(
    count: int,
    lower: Sequence[float],
    upper: Sequence[float],
    nvel: int,
    velocity_bases: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a quiet start: low-discrepancy positions and standard normal velocities.

    Parameters
    ----------
    count : int
        The number of markers.
    lower, upper : sequence of float
        The corners of the box, one per position axis.
    nvel : int
        The velocity components.
    velocity_bases : sequence of int, optional
        The primes of the velocity components' radical inverses, other than those of
        the positions (2, 3, ... for the axes after the first). The default takes the
        primes that follow; another species on the same positions takes others to get
        other velocities.

    Returns
    -------
    positions : numpy.ndarray, shape (ndim, count)
        Uniform in the box, a Hammersley set: with ``count = 2^m`` and a power-of-two
        grid in 2D, the same number in every cell.
    velocities : numpy.ndarray, shape (nvel, count)
        Standard normal, by the inverse CDF of the radical inverses; multiply by the
        thermal speed (and add a drift).
    """
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    ndim = lower_array.size
    position_bases = _PRIMES[: ndim - 1]
    if velocity_bases is None:
        velocity_bases = _PRIMES[ndim - 1 : ndim - 1 + nvel]
    velocity_bases = tuple(int(base) for base in velocity_bases)
    bases = position_bases + velocity_bases
    if len(velocity_bases) != nvel or len(set(bases)) != len(bases):
        raise ValueError(
            f"velocity_bases must be {nvel} primes other than {position_bases} "
            "and each other"
        )
    indices = np.arange(count)
    unit = [(indices + 0.5) / count]
    for base in bases:
        digits = max(1, int(np.ceil(np.log(max(count, 2)) / np.log(base) - 1e-12)))
        unit.append(radical_inverse(indices, base) + 0.5 * float(base) ** -digits)
    positions = (
        lower_array[:, None]
        + np.array(unit[:ndim]) * (upper_array - lower_array)[:, None]
    )
    return positions, ndtri(np.array(unit[ndim:]).reshape(nvel, count))


__all__ = [
    "ImportanceSampling",
    "MEAN_BIAS_SAMPLES",
    "PositionBias",
    "as_generator",
    "flux_speeds",
    "make_generator",
    "maxwellian_velocities",
    "normal",
    "quiet_start",
    "radical_inverse",
    "uniform",
]
