"""The NumPy version of the ``counting_sort`` kernel, run when Pyccel is unavailable.

Same arguments and result as ``counting_sort_pyccel.py``, with a stable argsort.
"""

from __future__ import annotations

import numpy as np


def counting_sort(
    keys: np.ndarray, alive: np.ndarray, starts: np.ndarray, order: np.ndarray
) -> None:
    """Write the stable order of the markers by key (see the Pyccel version)."""
    num_keys = starts.shape[0] - 2
    alive_keys = keys[alive]
    if alive_keys.size and (alive_keys.min() < 0 or alive_keys.max() >= num_keys):
        starts[0] = -1
        return
    binned = np.where(alive, keys, num_keys)
    order[...] = np.argsort(binned, kind="stable")
    starts[0] = 0
    starts[1:] = np.cumsum(np.bincount(binned, minlength=num_keys + 1))
