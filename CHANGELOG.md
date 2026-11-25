# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- `ParticleStore.sort_by_key(keys, num_keys=...)`: with the number of keys (cell indices), a
  counting sort (the new host kernel `counting_sort`, Pyccel and NumPy), O(markers + keys), that
  returns where each key starts. The sort gathers into a second set of storage arrays, kept
  between sorts, and swaps it in, instead of allocating new arrays and copying them back.
  A sort of 0.4M markers by 32^3 cells: 11.7 to 8.2 ms on one core, the counting sort itself
  2 ms against 4.4 ms for the radix argsort.

### Fixed
- The `mpiparticles.relativity` helpers return arrays of the kind they are given: a NumPy
  array stays a NumPy array with the CuPy backend (they used `xp.asarray`, which moved it to
  the device), so host codes such as beamtracer can use them under CuPy.

### Added
- Relativistic Boris push: `boris_push(..., relativistic=True, speed_of_light=...)` and the same
  option of `boris_velocity` push the proper velocity `u = γv` (stored in `store.velocities`)
  with the relativistic Boris scheme: the electric half kicks change `u`, the rotation is by
  `q/m B dt / γ` with `γ` after the first half kick, and the positions move with `u / γ`. It
  keeps `|u|` exactly in a magnetic field, gyrates at `ω_c / γ`, keeps every marker slower than
  light and is the Boris step for `|u| ≪ c`. `mpiparticles.relativity` has `SPEED_OF_LIGHT`,
  `lorentz_factor`, `proper_velocity`, `coordinate_velocity` and `kinetic_energy`. The blocks,
  for fused pushes of other codes too: `mpiparticles_relativistic_boris` and
  `mpiparticles_lorentz_factor` in `boris.cuh`, `relativistic_boris_update` and
  `lorentz_factor` in `mpiparticles.kernels.boris` (NumPy/CuPy, and
  `RELATIVISTIC_BORIS_METAL_HEADER`) and in `boris_pyccel.py` (`@inline`, no locals); the
  kernels `push_relativistic_boris` and `relativistic_boris_velocity` (Pyccel, NumPy, CUDA,
  Metal). Tests of every backend against the Pyccel blocks, gyration, hyperbolic motion
  (second order), the relativistic E×B drift and the non-relativistic limit; a tutorial,
  "Relativistic markers".
- Guiding centres (drift-kinetic markers): `guiding_centre_push` (a `ParticleStore`) and
  `guiding_centre_step` (plain arrays) move the guiding centre and the parallel velocity by a
  kick and a drift (E×B, grad-B and curvature drifts, parallel electric and mirror forces, the
  magnetic moment kept): the kick of `v_par` at the start, then the midpoint rule for the
  drift. Along the field this is the symplectic kick-drift of a particle push (bounded energy,
  no spurious damping when coupled to a field solve) and store the velocity rebuilt in the new field, the gyrophase carried
  along; the fields come as a function of the positions (E, B, grad |B|, curvature).
  `magnetic_moment` gives `mu`. The per-marker arithmetic is two blocks in every backend, for
  fused pushes of other codes too: `mpiparticles/kernels/guiding_centre.cuh` (CUDA),
  `mpiparticles.kernels.guiding_centre` (NumPy/CuPy, and `GUIDING_CENTRE_METAL_HEADER`) and
  `guiding_centre_pyccel.py` (`@inline` `gc_rates`, `gc_velocity`, written by
  `_generate_guiding_centre_pyccel.py`), with the kernels `guiding_centre_rates` and
  `guiding_centre_velocity`. Tests of every backend against the Pyccel blocks, the drifts
  against their formulas, a mirror bounce with bounded energy, the invariant of the kick-drift
  map; a tutorial section.
- Tests that the hand-written CUDA (`boris.cuh`, on cunumpy's emulator), NumPy and Metal Boris
  updates compute what the Pyccel block `boris_pyccel.boris_update` computes, run in Python:
  the Pyccel block is the source of truth of the four versions.
- `mpiparticles/kernels/boris_pyccel.py`: the Boris update for Pyccel kernels, `@inline`
  `boris_update(vx, vy, vz, kx, ky, kz, tx, ty, tz)`, with the arithmetic of `boris.cuh`. A
  Pyccel kernel of any package imports it and cunumpy's kernel cache builds it with the kernel
  (needs cunumpy with dependency builds); Pyccel inlines it. It has no local variables, so it is
  safe inside `#$ omp parallel for` loops. `push_boris` and `boris_velocity` use it.

- The Boris velocity update as shared building blocks for fused pushes:
  `mpiparticles/kernels/boris.cuh` (the CUDA device function `mpiparticles_boris`, for
  `#include` with `include_dirs=xp.kernels.source_roots("mpiparticles")`), and in
  `mpiparticles.kernels.boris` the NumPy/CuPy version `boris_update(velocities, kick, t)` and
  the Metal one, `BORIS_METAL_HEADER`. mpiparticles' own CUDA, NumPy and Metal kernels use
  them; the Pyccel kernels keep the arithmetic inline.
- Metal versions (Apple-silicon GPU, float32) of the `push_boris` and `boris_velocity`
  kernels, and `mpiparticles.kernels.catalog` (`KERNELS`, `metal_kernel(name)`) as in the other
  packages; the `metal` extra installs MLX. Tests of the CUDA versions on cunumpy's emulator,
  and of `boris_push` and `boris_velocity` on CuPy arrays (cunumpy's fake CuPy).
- A live demo in the docs ("Live demo" under Getting started): two species drifting in
  crossed electric and magnetic fields, pushed with `boris_push` in the reader's browser
  (Pyodide, on the NumPy kernels), and a Pyodide workflow that imports the package and runs the
  demo there. `pyccel` is no longer a dependency in the browser (`sys_platform != "emscripten"`),
  where it cannot be installed; the kernels run their NumPy versions there.
- `boris_push` takes `dt` and `charge_over_mass` as one number or one value per stored marker
  (several species in one store, a charge state per marker, a step per marker), and
  `boris_velocity(velocities, e_field, b_field, dt, charge_over_mass)` is its velocity update
  alone on a plain `(3, N)` array, for codes that move the positions themselves (a new kernel,
  `mpiparticles.kernels.boris_velocity`). The tutorial "Pushing markers" shows both.
- `boris_push(store, e_field, b_field, dt, charge_over_mass)`: one Boris step of the stored
  markers in electric and magnetic fields given at them, in 1D3V, 2D3V and 3D3V. A compiled
  kernel, `mpiparticles.kernels.push_boris` (Pyccel on the CPU, with NumPy and CUDA versions),
  the first of mpiparticles' kernels; `pyccel` is a dependency now. A tutorial, "Pushing markers".
- `velocity_moments(velocities, weights, max_order=2, comm=None)`: the sums
  `sum w v_i v_j ...` of every order up to `max_order` over the markers of all ranks;
  `moment_indices` gives their order (each symmetric moment once), `moment_name` their letters
  and `moment_values` the per-marker products, which a deposit turns into moments on a grid.
- `phase_space_density(positions, velocities, weights, axes, comm=None)` and `BinAxis`: the
  weights binned on one to three phase-space axes over the bin volume, summed over the ranks
  (moved from picnic's `PhaseSpaceBinning`). A tutorial, "Moments and binning".
- `ParticleStore`: per-rank marker storage in component-major arrays (positions, velocities,
  weights, alive, int64 ids) on NumPy or CuPy, with `reserve`, `add`, `deactivate`, `remove`,
  `clear`, `compact`, `compact_if_needed`, `sort_by_key`, `snapshot`/`restore` and a plasmacoll
  `MarkerSet` view (`as_marker_set`).
- `IdAllocator`: globally unique int64 ids with a rank stride, independent of the decomposition.
- `Decomposition` protocol and the reference `BoxDecomposition`.
- `migrate` (collective, sends only the leavers, ids as int64), `keep_owned`, `global_count` and
  `gather_columns`.
- Tests that launch `mpiexec -n 2` and `-n 4` jobs, a tutorial ("Moving markers between ranks"),
  guides and an example.
- Opt-in scope-profiler regions (`mpiparticles.migrate`, `mpiparticles.compact`, ...) tagged
  `"mpiparticles"`: off unless the application enables the tag, for example with
  `scope_profiler.setup(enable_tags=["mpiparticles"])`. scope-profiler (>= 0.7.5) is a new
  dependency.

### Changed

- `__version__` is read from the installed package metadata, not hardcoded.
- Docstrings follow the NumPy convention.
- `ParticleStore.sort_by_key` sorts and gathers with `xp.algorithms.sort_by_key(..., axis=-1)`
  (cunumpy >= 0.7.1 gathers component-major arrays row by row) instead of its own row loop; the
  dead markers sort last behind a key above every alive one.

### Removed

- The template's console entry point, example tutorial, example script and `setup_project.sh`.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
