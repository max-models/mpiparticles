"""The quiet start: a Hammersley set in position and velocity.

The positions are even over the cells (exactly, for powers of two in 2D),
the velocities standard normal with their moments to the order of ``1 / N``
rather than the ``1 / sqrt(N)`` of random draws, and none of the coordinates
is 0 or 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpiparticles.sampling import quiet_start, radical_inverse


def test_the_radical_inverse_mirrors_the_digits() -> None:
    np.testing.assert_allclose(
        radical_inverse(np.arange(8), 2), [0, 4, 2, 6, 1, 5, 3, 7] / np.float64(8)
    )
    np.testing.assert_allclose(
        radical_inverse(np.arange(1, 5), 3), [1 / 3, 2 / 3, 1 / 9, 4 / 9]
    )


def test_every_cell_of_a_power_of_two_grid_has_the_same_markers() -> None:
    positions, velocities = quiet_start(1 << 12, (0.0, -1.0), (16.0, 3.0), nvel=2)
    cells = np.floor(
        (positions - np.array([[0.0], [-1.0]])) / np.array([[1.0], [0.25]])
    )
    counts = np.bincount((cells[0] * 16 + cells[1]).astype(int), minlength=256)
    assert np.all(counts == 16)
    assert np.all(np.isfinite(velocities))
    # the moments of a standard normal, closer than 1 / sqrt(N) = 0.016
    # (measured: means 0.003 and 0.004, the base-3 and -5 sets being incomplete)
    assert np.max(np.abs(velocities.mean(axis=1))) < 5e-3
    np.testing.assert_allclose(velocities.var(axis=1), 1.0, atol=1e-2)
    assert abs(np.corrcoef(velocities)[0, 1]) < 1e-2
    assert abs(np.corrcoef(positions[1], velocities[0])[0, 1]) < 1e-2


def test_other_velocity_bases_give_other_velocities_on_the_same_positions() -> None:
    count = 3**7  # 0.5 + the base-3 lattice would hit 0 here; distinct primes do not
    positions, velocities = quiet_start(count, (0.0, 0.0), (1.0, 1.0), nvel=3)
    again, others = quiet_start(
        count, (0.0, 0.0), (1.0, 1.0), nvel=3, velocity_bases=(7, 11, 13)
    )
    np.testing.assert_array_equal(positions, again)
    assert np.all(np.isfinite(others))
    assert abs(np.corrcoef(velocities[0], others[0])[0, 1]) < 0.05
    one_d, speeds = quiet_start(8, (0.0,), (8.0,), nvel=1)
    np.testing.assert_allclose(one_d[0], np.arange(8) + 0.5)
    # the bit-reversed quantiles of the 1D quiet start
    assert np.argsort(np.argsort(speeds[0])).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]


def test_the_bases_are_checked() -> None:
    with pytest.raises(ValueError, match="velocity_bases"):
        quiet_start(16, (0.0, 0.0), (1.0, 1.0), nvel=2, velocity_bases=(2, 5))
    with pytest.raises(ValueError, match="velocity_bases"):
        quiet_start(16, (0.0, 0.0), (1.0, 1.0), nvel=2, velocity_bases=(5,))
