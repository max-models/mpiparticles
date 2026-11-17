"""The live demo of the docs: the E×B drift of two species, moved with mpiparticles' Boris push.

It runs in the browser (Pyodide) on the NumPy versions of the kernels; `run` takes the values
of the sliders and returns the figures. Normalized units: the charge over the mass is +1 for
one species and -4 for the other, the thermal speed is 1.
"""

import numpy as np

from mpiparticles import ParticleStore, boris_push, boris_velocity

#: The markers of each species, and how many of them draw their trajectory.
MARKERS = 100
TRACKED = 3
#: The charge over the mass of the two species.
SPECIES = {"q/m = +1": 1.0, "q/m = −4": -4.0}
COLORS = ("#1f77b4", "#d62728")


def run(e_field=0.5, b_field=1.0, dt=0.1, steps=150):
    """Push both species through uniform fields and measure their drift.

    Parameters
    ----------
    e_field : float
        The electric field along y.
    b_field : float
        The magnetic field along z.
    dt : float
        The time step.
    steps : int
        The number of Boris steps.

    Returns
    -------
    dict
        Two Plotly figures (the trajectories, the mean guiding centre against time) and the
        measured drift speeds next to E/B.
    """
    steps = int(steps)
    rng = np.random.default_rng(1)
    count = MARKERS * len(SPECIES)
    charge_over_mass = np.repeat(list(SPECIES.values()), MARKERS)
    positions = rng.uniform([[0.0], [0.0]], [[2.0], [4.0]], size=(2, count))
    velocities = rng.normal(0.0, 1.0, size=(3, count))

    e = np.array([0.0, e_field, 0.0])
    b = np.array([0.0, 0.0, b_field])
    store = ParticleStore(ndim=2, nvel=3)
    store.add(positions, velocities)
    # the leapfrog starts with the velocities half a step back
    boris_velocity(store.velocities[:, :count], e, b, -0.5 * dt, charge_over_mass)

    tracked = np.r_[np.arange(TRACKED), MARKERS + np.arange(TRACKED)]
    paths = np.empty((steps + 1, 2, tracked.size))
    centre = np.empty((steps + 1, count))
    omega = charge_over_mass * b_field  # signed gyrofrequency
    for step in range(steps + 1):
        if step > 0:
            boris_push(store, e, b, dt, charge_over_mass)
        x = store.positions[:, :count]
        v = store.velocities[:, :count]
        paths[step] = x[:, tracked]
        # guiding centre along x: x + (v × z)_x / omega_c
        centre[step] = x[0] + v[1] / omega

    t = dt * np.arange(steps + 1)
    trajectories, means, numbers = [], [], {"E/B": e_field / b_field}
    for index, (name, color) in enumerate(zip(SPECIES, COLORS, strict=True)):
        members = slice(index * MARKERS, (index + 1) * MARKERS)
        final = store.positions[:, members]
        trajectories.append(
            {
                "x": final[0],
                "y": final[1],
                "mode": "markers",
                "marker": {"size": 4, "color": color, "opacity": 0.5},
                "name": f"{name}, at the end",
            }
        )
        for k in range(TRACKED):
            path = paths[:, :, index * TRACKED + k]
            trajectories.append(
                {
                    "x": path[0],
                    "y": path[1],
                    "mode": "lines",
                    "line": {"color": color, "width": 1.5},
                    "name": name,
                    "legendgroup": name,
                    "showlegend": k == 0,
                }
            )
        mean = centre[:, members].mean(axis=1)
        means.append(
            {
                "x": t,
                "y": mean - mean[0],
                "mode": "lines",
                "name": name,
                "line": {"color": color},
            }
        )
        numbers[f"Drift, {name}"] = (
            float(np.polyfit(t, mean, 1)[0]) if steps > 0 else 0.0
        )
    means.append(
        {
            "x": t,
            "y": e_field / b_field * t,
            "mode": "lines",
            "name": "(E/B) t",
            "line": {"color": "gray", "dash": "dash"},
        }
    )
    numbers["ω<sub>c</sub>Δt, q/m = −4"] = 4.0 * b_field * dt

    return {
        "figure": [
            {
                "data": trajectories,
                "layout": {
                    "xaxis": {"title": {"text": "x"}},
                    "yaxis": {"title": {"text": "y"}, "scaleanchor": "x"},
                },
            },
            {
                "data": means,
                "layout": {
                    "xaxis": {"title": {"text": "time"}},
                    "yaxis": {"title": {"text": "mean guiding-centre shift along x"}},
                },
            },
        ],
        "numbers": numbers,
    }
