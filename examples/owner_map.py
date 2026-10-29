"""Marker owners.

Which rank owns which marker: split a box between four ranks with a BoxDecomposition and colour random markers by their
owner. Four ranks are simulated by a stand-in communicator, so the script runs without MPI;
under a real run each rank would only ask about its own markers.
"""

import cunumpy as xp
import matplotlib.pyplot as plt
import numpy as np

from mpiparticles import BoxDecomposition


class Rank:
    """A communicator that only knows its rank and size."""

    def __init__(self, rank, size):
        self.rank, self.size = rank, size

    def Get_rank(self):
        """Return the rank."""
        return self.rank

    def Get_size(self):
        """Return the number of ranks."""
        return self.size


decomposition = BoxDecomposition(Rank(0, 4), [0.0, 0.0], [1.0, 1.0], [2, 2])
positions = np.random.default_rng(3).uniform(-0.1, 1.1, size=(2, 600))
owners = np.asarray(xp.to_numpy(decomposition.owners(xp.asarray(positions))))

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(*positions, c=owners, cmap="tab10", vmin=0, vmax=9, s=10)
ax.axvline(0.5, color="k", lw=0.8)
ax.axhline(0.5, color="k", lw=0.8)
ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ls="--"))
ax.set(xlabel="x", ylabel="y", title="Owner rank (outside the box: nearest block)")
ax.set_aspect("equal")
