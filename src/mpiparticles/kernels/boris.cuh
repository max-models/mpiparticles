// The Boris velocity update as a device function, and its relativistic version, for the CUDA
// kernels of mpiparticles and of the codes that fuse them into their own pushes:
//
//     #include "mpiparticles/kernels/boris.cuh"
//
// with include_dirs=xp.kernels.source_roots("mpiparticles"). boris.py next to this file has
// the NumPy version (mpiparticles.kernels.boris.boris_update) and the Metal one
// (BORIS_METAL_HEADER, RELATIVISTIC_BORIS_METAL_HEADER) and boris_pyccel.py the Pyccel one.

#pragma once

// The velocity (vx, vy, vz) after one Boris step, in place: the half kick (kx, ky, kz),
// q/m E dt/2, the rotation about t = q/m B dt/2 and the second half kick.
__device__ inline void mpiparticles_boris(double& vx, double& vy, double& vz,
                                          double kx, double ky, double kz,
                                          double tx, double ty, double tz) {
    vx += kx;
    vy += ky;
    vz += kz;
    const double factor = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
    const double sx = factor * tx, sy = factor * ty, sz = factor * tz;
    const double px = vx + vy * tz - vz * ty;
    const double py = vy + vz * tx - vx * tz;
    const double pz = vz + vx * ty - vy * tx;
    vx += py * sz - pz * sy;
    vy += pz * sx - px * sz;
    vz += px * sy - py * sx;
    vx += kx;
    vy += ky;
    vz += kz;
}

// The Lorentz factor gamma = sqrt(1 + |u|^2 / c^2) of the proper velocity u = gamma v;
// inv_c2 is 1 / c^2.
__device__ inline double mpiparticles_lorentz_factor(double ux, double uy, double uz,
                                                     double inv_c2) {
    return sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * inv_c2);
}

// The proper velocity (ux, uy, uz) = gamma v after one relativistic Boris step, in place:
// the half kick (kx, ky, kz), q/m E dt/2, the rotation about t = q/m B dt/(2 gamma) with
// gamma that of u + k, and the second half kick. (bx, by, bz) is q/m B dt/2, inv_c2 is 1/c^2.
__device__ inline void mpiparticles_relativistic_boris(double& ux, double& uy, double& uz,
                                                       double kx, double ky, double kz,
                                                       double bx, double by, double bz,
                                                       double inv_c2) {
    const double inv_gamma =
        1.0 / mpiparticles_lorentz_factor(ux + kx, uy + ky, uz + kz, inv_c2);
    mpiparticles_boris(ux, uy, uz, kx, ky, kz, bx * inv_gamma, by * inv_gamma,
                       bz * inv_gamma);
}
