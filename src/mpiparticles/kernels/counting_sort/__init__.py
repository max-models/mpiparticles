"""The counting_sort kernel: Pyccel and NumPy versions with the same arguments.

A host-only kernel (no CUDA or Metal version): on CuPy arrays the store sorts with
`cunumpy.algorithms.sort_by_key`, a radix sort. ``kernel(...)`` runs the Pyccel version,
or the NumPy version if Pyccel is unavailable.
"""

from mpiparticles.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
