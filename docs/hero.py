"""The figure on the docs home page: markers circling over four ranks and migrating between them.

A box split over a 2 x 2 grid of MPI ranks (:class:`mpiparticles.BoxDecomposition`) holds
markers in a :class:`mpiparticles.ParticleStore` on each rank. They circle a vortex, the inner
rings faster, and after every push :func:`mpiparticles.migrate` sends the markers that left a rank
to their new owner. The owner of each marker at every step is the store it was found in
(collected with :func:`mpiparticles.gather_columns`); the dots in
``docs/src/assets/hero.svg`` (drawn with ``plasma_plots.hero``) change colour when they change
rank. Rerun with::

    mpiexec -n 4 python docs/hero.py   # or: make docs-hero
"""

from pathlib import Path

import cunumpy as xp
import maybempi
import numpy as np
from plasma_plots.hero import HeroFigure

from mpiparticles import (
    BoxDecomposition,
    IdAllocator,
    ParticleStore,
    gather_columns,
    migrate,
)

comm = maybempi.get_mpi().COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
lower, upper, centre = np.array([3.4, 0.0]), np.array([8.0, 3.0]), np.array([5.7, 1.5])
decomposition = BoxDecomposition(
    comm, lower, upper, proc_sizes=[2, 2] if size == 4 else [size, 1]
)
store = ParticleStore(ndim=2, nvel=2, id_allocator=IdAllocator.from_comm(comm))

# rings of markers around the centre; ring k turns turns[k] times per loop
radii, turns, per_ring = np.array([0.35, 0.65, 0.95, 1.25]), np.array([3, 2, 2, 1]), 7
if rank == 0:
    ring = np.repeat(np.arange(len(radii)), per_ring)
    phase = 2 * np.pi * (np.arange(ring.size) / per_ring + 0.37 * ring)
    start = centre[:, None] + radii[ring] * np.stack([np.cos(phase), np.sin(phase)])
    store.add(xp.asarray(start), xp.zeros((2, ring.size)), ids=xp.arange(ring.size))
migrate(store, decomposition)

steps, owners, positions = 240, [], []
for _ in range(steps):
    relative = store.active_positions() - xp.asarray(centre[:, None])
    radius = xp.hypot(*relative)
    angle = xp.arctan2(relative[1], relative[0])
    angle = (
        angle
        + 2
        * np.pi
        / steps
        * xp.asarray(turns)[
            xp.argmin(xp.abs(radius[None] - xp.asarray(radii[:, None])), axis=0)
        ]
    )
    store.set_active_positions(
        xp.asarray(centre[:, None]) + radius * xp.stack([xp.cos(angle), xp.sin(angle)])
    )
    migrate(
        store, decomposition
    )  # the markers that crossed a rank border move to their owner
    store.compact_if_needed()
    record = gather_columns(store.active_ids(), comm)
    where = gather_columns(xp.full(store.num_alive, rank), comm)
    place = gather_columns(store.active_positions(), comm)
    if rank == 0:
        order = np.argsort(record)
        owners.append(where[order])
        positions.append(place[:, order])

if rank == 0:
    owners, positions = (
        np.array(owners),
        np.array(positions),
    )  # (steps, n), (steps, 2, n)
    fig = HeroFigure(xlim=(0, 8), ylim=(0, 3))
    mid = (lower + upper) / 2
    corners = [
        (lower[0], lower[1], mid[0], mid[1]),
        (lower[0], mid[1], mid[0], upper[1]),
        (mid[0], lower[1], upper[0], mid[1]),
        (mid[0], mid[1], upper[0], upper[1]),
    ]
    for r, (x0, y0, x1, y1) in enumerate(corners[:size] if size == 4 else []):
        fig.rect(
            x0,
            y0,
            x1,
            y1,
            tone="low",
            opacity=0.7 if r in (0, 3) else 0.3,
            outline="muted",
            title=f"rank {r}",
        )
    t = np.linspace(0, 2 * np.pi, 120)
    for r in radii:
        fig.line(
            centre[0] + r * np.cos(t),
            centre[1] + r * np.sin(t),
            tone="muted",
            width=1,
            opacity=0.35,
            dash="2 6",
        )
    duration = 16.0
    tones = np.array(["f-accent", "f-high", "f-high", "f-accent"])
    for i in range(positions.shape[2]):
        x, y = positions[::2, 0, i], positions[::2, 1, i]
        x, y = np.r_[x[-1], x], np.r_[y[-1], y]  # the loop ends where it starts
        dot = fig.tracer(x, duration, r=5.0, tone="high", y=y)
        dot.animate(
            "class", tones[np.r_[owners[-1, i], owners[:, i]] % 4][::2], duration
        )
    fig.save(Path(__file__).parent / "src" / "assets" / "hero.svg")
