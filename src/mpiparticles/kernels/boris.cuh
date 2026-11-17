// The Boris velocity update as a device function, for the CUDA kernels of mpiparticles and
// of the codes that fuse it into their own pushes:
//
//     #include "mpiparticles/kernels/boris.cuh"
//
// with include_dirs=xp.kernels.source_roots("mpiparticles"). boris.py next to this file has
// the NumPy version (mpiparticles.kernels.boris.boris_update) and the Metal one
// (BORIS_METAL_HEADER); the Pyccel kernels have the same arithmetic inline.

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
