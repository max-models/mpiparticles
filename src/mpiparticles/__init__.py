"""Distributed marker (particle) container for MPI codes.

A :class:`ParticleStore` holds the markers of one rank in component-major
arrays (CPU or GPU through ``cunumpy``) with stable int64 ids; a
:class:`Decomposition` says which rank owns which position; :func:`migrate`
moves the markers that left a rank's subdomain to their owners; :func:`boris_push`
advances them through given electric and magnetic fields (1D3V, 2D3V and 3D3V), and
:func:`guiding_centre_push` moves them as guiding centres; with ``relativistic=True`` the
Boris push advances proper velocities (`mpiparticles.relativity`). `mpiparticles.sampling`
draws markers (Maxwellian and flux speeds, quiet starts, :class:`ImportanceSampling`) and
`mpiparticles.injection` has the timestep sources that emit them each step. Without an MPI
launcher everything runs serially, with no MPI calls.

The expensive operations are scope-profiler regions named ``mpiparticles.<operation>``,
tagged ``"mpiparticles"`` and off by default; enable them with
``scope_profiler.setup(enable_tags=["mpiparticles"])``.

Example
-------
>>> import cunumpy as xp
>>> from mpiparticles import BoxDecomposition, ParticleStore, migrate
>>> store = ParticleStore(ndim=1, nvel=3)
>>> _ = store.add(xp.asarray([[0.25, 0.75]]), xp.zeros((3, 2)))
>>> deco = BoxDecomposition(None, [0.0], [1.0], [1])
>>> migrate(store, deco)
(0, 0)
"""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from mpiparticles._profiling import PROFILE_TAGS
from mpiparticles.collective import gather_columns, global_count
from mpiparticles.decomposition import BoxDecomposition, Decomposition
from mpiparticles.exchange import keep_owned, migrate
from mpiparticles.guiding_centre import (
    guiding_centre_push,
    guiding_centre_step,
    magnetic_moment,
)
from mpiparticles.ids import IdAllocator
from mpiparticles.injection import MarkerSpecies
from mpiparticles.moments import (
    BinAxis,
    bin_axes,
    moment_indices,
    moment_name,
    moment_values,
    phase_space_density,
    velocity_moments,
)
from mpiparticles.push import boris_push, boris_velocity
from mpiparticles.relativity import (
    SPEED_OF_LIGHT,
    coordinate_velocity,
    kinetic_energy,
    lorentz_factor,
    proper_velocity,
)
from mpiparticles.sampling import ImportanceSampling, PositionBias, quiet_start
from mpiparticles.store import MarkerView, ParticleStore, StoreSnapshot

# The profiling regions are off unless the application enables the tag.
sp.register_opt_in_tag(*PROFILE_TAGS)

try:
    __version__ = version("mpiparticles")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BinAxis",
    "BoxDecomposition",
    "Decomposition",
    "IdAllocator",
    "ImportanceSampling",
    "MarkerSpecies",
    "MarkerView",
    "ParticleStore",
    "PositionBias",
    "SPEED_OF_LIGHT",
    "StoreSnapshot",
    "bin_axes",
    "boris_push",
    "boris_velocity",
    "coordinate_velocity",
    "gather_columns",
    "global_count",
    "guiding_centre_push",
    "guiding_centre_step",
    "keep_owned",
    "kinetic_energy",
    "lorentz_factor",
    "magnetic_moment",
    "migrate",
    "moment_indices",
    "moment_name",
    "moment_values",
    "phase_space_density",
    "proper_velocity",
    "quiet_start",
    "velocity_moments",
]
