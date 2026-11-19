// The CUDA version (GPU) of the guiding_centre_rates kernel; guiding_centre_rates_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments. One thread per
// marker.
//
// dX/dt into drift (3, N) and dv_par/dt into accel (N,): the fields at the markers e_field,
// b_field, grad_b and curvature (3, N), v_par and mu (N,), the charge and the mass.

#include "cunumpy/array_view.cuh"
#include "mpiparticles/kernels/guiding_centre.cuh"

extern "C" __global__
void guiding_centre_rates(
    Array2D<double> e_field,
    Array2D<double> b_field,
    Array2D<double> grad_b,
    Array2D<double> curvature,
    Array1D<double> v_par,
    Array1D<double> mu,
    double charge,
    double mass,
    Array2D<double> drift,
    Array1D<double> accel
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= v_par.shape[0]) return;
    double dx, dy, dz, dv;
    mpiparticles_gc_rates(
        e_field(0, marker), e_field(1, marker), e_field(2, marker),
        b_field(0, marker), b_field(1, marker), b_field(2, marker),
        grad_b(0, marker), grad_b(1, marker), grad_b(2, marker),
        curvature(0, marker), curvature(1, marker), curvature(2, marker),
        v_par(marker), mu(marker), charge, mass, dx, dy, dz, dv);
    drift(0, marker) = dx;
    drift(1, marker) = dy;
    drift(2, marker) = dz;
    accel(marker) = dv;
}
