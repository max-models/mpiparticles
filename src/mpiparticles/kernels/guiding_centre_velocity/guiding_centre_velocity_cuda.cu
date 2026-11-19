// The CUDA version (GPU) of the guiding_centre_velocity kernel;
// guiding_centre_velocity_pyccel.py next to this file is the Pyccel version (CPU), with the
// same arguments. One thread per marker.
//
// The velocities (3, N) to store after a guiding-centre step, in place: b_old and b_new
// (3, N) the field at the markers before and after it, v_par and mu (N,), the mass.

#include "cunumpy/array_view.cuh"
#include "mpiparticles/kernels/guiding_centre.cuh"

extern "C" __global__
void guiding_centre_velocity(
    Array2D<double> velocities,
    Array2D<double> b_old,
    Array2D<double> b_new,
    Array1D<double> v_par,
    Array1D<double> mu,
    double mass
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= v_par.shape[0]) return;
    double vx = velocities(0, marker), vy = velocities(1, marker), vz = velocities(2, marker);
    mpiparticles_gc_velocity(
        vx, vy, vz,
        b_old(0, marker), b_old(1, marker), b_old(2, marker),
        b_new(0, marker), b_new(1, marker), b_new(2, marker),
        v_par(marker), mu(marker), mass);
    velocities(0, marker) = vx;
    velocities(1, marker) = vy;
    velocities(2, marker) = vz;
}
