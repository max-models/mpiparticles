// The CUDA version (GPU) of the push_relativistic_boris kernel;
// push_relativistic_boris_pyccel.py next to this file is the Pyccel version (CPU), with the
// same arguments. One thread per marker.
//
// One relativistic Boris step: positions (ndim, N) with ndim 1, 2 or 3; proper_velocities
// (u = gamma v), e_field and b_field (3, N), the fields at the markers at the middle of the step.

#include "cunumpy/array_view.cuh"
#include "mpiparticles/kernels/boris.cuh"

extern "C" __global__
void push_relativistic_boris(
    Array2D<double> positions,
    Array2D<double> proper_velocities,
    Array2D<double> e_field,
    Array2D<double> b_field,
    double charge_over_mass,
    double dt,
    double speed_of_light
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    const double half = 0.5 * charge_over_mass * dt;
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
    const double step = dt / mpiparticles_lorentz_factor(ux, uy, uz, inv_c2);
    for (long long d = 0; d < positions.shape[0]; ++d) {
        positions(d, marker) += step * proper_velocities(d, marker);
    }
}
