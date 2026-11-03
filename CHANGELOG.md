# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
