"""mpiparticles' kernels by name, and their Metal versions.

``KERNELS[name]`` is the kernel of ``mpiparticles/kernels/<name>/``: CUDA for CuPy
arrays, Pyccel (or NumPy) on the host. :func:`metal_kernel` returns the Metal version
(Apple-silicon GPU, float32), a function with the arguments of the Pyccel version.

Marker arrays are component-major: positions ``(ndim, N)``, velocities and fields at
the markers ``(3, N)``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import cunumpy as xp
from cunumpy.kernels import metal_available

__all__ = ["KERNELS", "KERNEL_NAMES", "METAL_KERNEL_NAMES", "metal_kernel"]

#: The kernels, by name.
KERNEL_NAMES = (
    "boris_velocity",
    "push_boris",
    "relativistic_boris_velocity",
    "push_relativistic_boris",
    "guiding_centre_rates",
    "guiding_centre_velocity",
)

_PACKAGE = __name__.rpartition(".")[0]

#: Every kernel of mpiparticles, by name.
KERNELS = xp.kernels.KernelCatalog(
    {
        name: importlib.import_module(f"{_PACKAGE}.{name}").kernel
        for name in KERNEL_NAMES
    }
)

#: The kernels with a Metal version (``<name>/<name>_metal.py``): all of them.
METAL_KERNEL_NAMES = KERNEL_NAMES


def metal_kernel(name: str) -> Callable[..., None]:
    """Return the Metal version of kernel ``name``: a function with the Pyccel arguments.

    Needs MLX and a Metal GPU (``pip install 'mpiparticles[metal]'``); the arrays are
    NumPy arrays, computed in float32 and written back into the float64 arrays.

    Parameters
    ----------
    name : str
        The kernel, one of `METAL_KERNEL_NAMES`.

    Returns
    -------
    callable
        The Metal version, with the arguments of the Pyccel version.

    Raises
    ------
    KeyError
        If the kernel has no Metal version.
    RuntimeError
        If there is no usable Metal GPU.
    """
    if name not in METAL_KERNEL_NAMES:
        raise KeyError(f"{name!r} has no Metal version; have {METAL_KERNEL_NAMES}")
    if not metal_available():
        raise RuntimeError(
            "Metal kernels need MLX and a Metal GPU (mpiparticles[metal])"
        )
    module = importlib.import_module(f"{_PACKAGE}.{name}.{name}_metal")
    return getattr(module, name)
