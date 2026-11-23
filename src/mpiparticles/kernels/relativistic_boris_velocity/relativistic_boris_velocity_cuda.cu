// The CUDA version (GPU) of the relativistic_boris_velocity kernel;
// relativistic_boris_velocity_pyccel.py next to this file is the Pyccel version (CPU), with
// the same arguments. One thread per marker.
//
// The relativistic Boris update: proper_velocities (u = gamma v), e_field and b_field (3, N),
// the fields at the markers at the middle of the step; charge_over_mass and dt (N,), one per
// marker.

#include "cunumpy/array_view.cuh"
#include "mpiparticles/kernels/boris.cuh"

extern "C" __global__
void relativistic_boris_velocity(
    Array2D<double> proper_velocities,
    Array2D<double> e_field,
    Array2D<double> b_field,
    Array1D<double> charge_over_mass,
    Array1D<double> dt,
    double speed_of_light
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= proper_velocities.shape[1]) return;
    const double half = 0.5 * charge_over_mass(marker) * dt(marker);
    const double inv_c2 = 1.0 / (speed_of_light * speed_of_light);
    double ux = proper_velocities(0, marker), uy = proper_velocities(1, marker),
           uz = proper_velocities(2, marker);
    mpiparticles_relativistic_boris(ux, uy, uz,
                                    half * e_field(0, marker), half * e_field(1, marker),
                                    half * e_field(2, marker),
                                    half * b_field(0, marker), half * b_field(1, marker),
                                    half * b_field(2, marker), inv_c2);
    proper_velocities(0, marker) = ux;
    proper_velocities(1, marker) = uy;
    proper_velocities(2, marker) = uz;
}
