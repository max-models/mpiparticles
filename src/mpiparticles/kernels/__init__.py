"""mpiparticles' compiled kernels, one folder per kernel.

Every kernel ``<name>`` lives in its own folder, with versions that take the same arguments::

    mpiparticles/kernels/<name>/__init__.py        kernel = KERNEL_PACKAGE.from_folder(__name__)
    mpiparticles/kernels/<name>/<name>_pyccel.py   CPU, compiled with Pyccel on first use
    mpiparticles/kernels/<name>/<name>_numpy.py    CPU, when Pyccel is unavailable
    mpiparticles/kernels/<name>/<name>_cuda.cu     GPU, compiled with NVRTC

``kernel(...)`` runs the CUDA version when the arrays are CuPy arrays and the compiled
Pyccel version otherwise (see `cunumpy.kernels.KernelPackage`). The Pyccel builds are
kept under ``MPIPARTICLES_KERNEL_CACHE`` (default ``~/.cache/mpiparticles/kernels``;
``MPIPARTICLES_OPENMP=0`` builds serial). Importing a kernel needs neither Pyccel nor CuPy.

Marker arrays are component-major, as in the store: positions ``(ndim, N)``, velocities and
fields at the markers ``(3, N)``.
"""

import cunumpy as xp

#: The build cache and launch options of mpiparticles' kernels.
KERNEL_PACKAGE = xp.kernels.KernelPackage("mpiparticles")
