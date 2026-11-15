"""The Boris velocity update, with a charge over mass and a time step per marker.

The Pyccel version (CPU) of the ``boris_velocity`` kernel; ``boris_velocity_cuda.cu`` next
to this file is the CUDA version (GPU), with the same arguments. Loaded by
`mpiparticles.kernels`; this file is compiled with Pyccel there.
"""


def boris_velocity(
    velocities: "float[:,:]",
    e_field: "float[:,:]",
    b_field: "float[:,:]",
    charge_over_mass: "float[:]",
    dt: "float[:]",
) -> None:
    """Advance the velocities by one Boris step, in place; the positions are not touched.

    ``velocities``, ``e_field`` and ``b_field`` are ``(3, N)``, the fields at the markers at
    the middle of the step; ``charge_over_mass`` and ``dt`` are ``(N,)``. The velocity takes
    half the electric kick, the magnetic rotation and the other half, as in ``push_boris``.
    """
    num_markers = velocities.shape[1]
    # fmt: off
    #$ omp parallel for private(half, vx, vy, vz, tx, ty, tz, factor, sx, sy, sz, px, py, pz) schedule(static)
    # fmt: on
    for marker in range(num_markers):
        half = 0.5 * charge_over_mass[marker] * dt[marker]
        # half the electric kick
        vx = velocities[0, marker] + half * e_field[0, marker]
        vy = velocities[1, marker] + half * e_field[1, marker]
        vz = velocities[2, marker] + half * e_field[2, marker]
        # the magnetic rotation: v' = v- + v- x t, v+ = v- + v' x s
        tx = half * b_field[0, marker]
        ty = half * b_field[1, marker]
        tz = half * b_field[2, marker]
        factor = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
        sx = factor * tx
        sy = factor * ty
        sz = factor * tz
        px = vx + vy * tz - vz * ty
        py = vy + vz * tx - vx * tz
        pz = vz + vx * ty - vy * tx
        vx = vx + py * sz - pz * sy
        vy = vy + pz * sx - px * sz
        vz = vz + px * sy - py * sx
        # the other half of the kick
        velocities[0, marker] = vx + half * e_field[0, marker]
        velocities[1, marker] = vy + half * e_field[1, marker]
        velocities[2, marker] = vz + half * e_field[2, marker]
