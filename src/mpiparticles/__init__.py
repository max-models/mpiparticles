"""Distributed marker (particle) container for MPI codes.

A :class:`ParticleStore` holds the markers of one rank in component-major
arrays (CPU or GPU through ``cunumpy``) with stable int64 ids; a
:class:`Decomposition` says which rank owns which position; :func:`migrate`
moves the markers that left a rank's subdomain to their owners. Without an MPI
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
from mpiparticles.ids import IdAllocator
from mpiparticles.moments import (
    BinAxis,
    moment_indices,
    moment_name,
    moment_values,
    phase_space_density,
    velocity_moments,
)
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
    "MarkerView",
    "ParticleStore",
    "StoreSnapshot",
    "gather_columns",
    "global_count",
    "keep_owned",
    "migrate",
    "moment_indices",
    "moment_name",
    "moment_values",
    "phase_space_density",
    "velocity_moments",
]
