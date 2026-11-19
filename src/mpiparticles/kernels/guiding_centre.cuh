// The guiding-centre blocks as device functions, for the CUDA kernels of mpiparticles and
// of the codes that fuse them into their own pushes:
//
//     #include "mpiparticles/kernels/guiding_centre.cuh"
//
// with include_dirs=xp.kernels.source_roots("mpiparticles"). guiding_centre.py next to this
// file has the NumPy versions and the Metal one (GUIDING_CENTRE_METAL_HEADER);
// guiding_centre_pyccel.py the Pyccel one, with the same arithmetic.

#pragma once

#include <math.h>

// dX/dt (dx, dy, dz) and dv_par/dt (dv) of a guiding centre: parallel motion, E x B,
// grad-B and curvature drifts; the parallel electric and mirror forces.
__device__ inline void mpiparticles_gc_rates(
    double ex, double ey, double ez,
    double bx, double by, double bz,
    double gx, double gy, double gz,
    double kx, double ky, double kz,
    double v_par, double mu, double charge, double mass,
    double& dx, double& dy, double& dz, double& dv) {
    const double magnitude = sqrt(bx * bx + by * by + bz * bz);
    const double ux = bx / magnitude, uy = by / magnitude, uz = bz / magnitude;
    const double exb = 1.0 / (magnitude * magnitude);
    const double grad = mu / (charge * magnitude);
    const double curve = mass * v_par * v_par / (charge * magnitude);
    dx = v_par * ux + (ey * bz - ez * by) * exb + grad * (uy * gz - uz * gy)
         + curve * (uy * kz - uz * ky);
    dy = v_par * uy + (ez * bx - ex * bz) * exb + grad * (uz * gx - ux * gz)
         + curve * (uz * kx - ux * kz);
    dz = v_par * uz + (ex * by - ey * bx) * exb + grad * (ux * gy - uy * gx)
         + curve * (ux * ky - uy * kx);
    dv = (charge / mass) * (ex * ux + ey * uy + ez * uz)
         - (mu / mass) * (ux * gx + uy * gy + uz * gz);
}

// The velocity (vx, vy, vz) to store after a step, in place: v_par along the new b, and
// v_perp = sqrt(2 mu B / m) along the old perpendicular velocity projected normal to it.
__device__ inline void mpiparticles_gc_velocity(
    double& vx, double& vy, double& vz,
    double ox, double oy, double oz,
    double nx, double ny, double nz,
    double v_par, double mu, double mass) {
    const double old_norm = sqrt(ox * ox + oy * oy + oz * oz);
    const double ax = ox / old_norm, ay = oy / old_norm, az = oz / old_norm;
    const double magnitude = sqrt(nx * nx + ny * ny + nz * nz);
    const double ux = nx / magnitude, uy = ny / magnitude, uz = nz / magnitude;
    const double along_old = vx * ax + vy * ay + vz * az;
    const double px = vx - along_old * ax, py = vy - along_old * ay, pz = vz - along_old * az;
    const double along_new = px * ux + py * uy + pz * uz;
    const double cx = px - along_new * ux, cy = py - along_new * uy, cz = pz - along_new * uz;
    const double norm = fmax(sqrt(cx * cx + cy * cy + cz * cz), 1.0e-300);
    const double v_perp = sqrt(fmax(2.0 * mu * magnitude / mass, 0.0));
    vx = v_par * ux + v_perp * cx / norm;
    vy = v_par * uy + v_perp * cy / norm;
    vz = v_par * uz + v_perp * cz / norm;
}
