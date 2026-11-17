// The CUDA version (GPU) of the push_boris kernel; push_boris_pyccel.py next to this file is
// the Pyccel version (CPU), with the same arguments. One thread per marker.
//
// One Boris step: positions (ndim, N) with ndim 1, 2 or 3; velocities, e_field and b_field
// (3, N), the fields at the markers at the middle of the step.

#include "cunumpy/array_view.cuh"
#include "mpiparticles/kernels/boris.cuh"

extern "C" __global__
void push_boris(
    Array2D<double> positions,
    Array2D<double> velocities,
    Array2D<double> e_field,
    Array2D<double> b_field,
    double charge_over_mass,
    double dt
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    const double half = 0.5 * charge_over_mass * dt;
    double vx = velocities(0, marker), vy = velocities(1, marker), vz = velocities(2, marker);
    mpiparticles_boris(vx, vy, vz,
                       half * e_field(0, marker), half * e_field(1, marker),
                       half * e_field(2, marker),
                       half * b_field(0, marker), half * b_field(1, marker),
                       half * b_field(2, marker));
    velocities(0, marker) = vx;
    velocities(1, marker) = vy;
    velocities(2, marker) = vz;
    for (long long d = 0; d < positions.shape[0]; ++d) {
        positions(d, marker) += dt * velocities(d, marker);
    }
}
