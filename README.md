# mpiparticles


<!-- README.md is generated from README.qmd: edit the .qmd and run `make readme`. -->

[![Tests](https://github.com/max-models/mpiparticles/actions/workflows/test_pytest.yml/badge.svg)](https://github.com/max-models/mpiparticles/actions/workflows/test_pytest.yml)
[![Static
analysis](https://github.com/max-models/mpiparticles/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/max-models/mpiparticles/actions/workflows/static_analysis.yml)
[![Docs](https://github.com/max-models/mpiparticles/actions/workflows/docs.yml/badge.svg)](https://max-models.github.io/mpiparticles/)
[![codecov](https://codecov.io/gh/max-models/mpiparticles/branch/main/graph/badge.svg)](https://codecov.io/gh/max-models/mpiparticles)
[![PyPI](https://img.shields.io/pypi/v/mpiparticles.png)](https://pypi.org/project/mpiparticles/)
[![Python](https://img.shields.io/pypi/pyversions/mpiparticles.png)](https://pypi.org/project/mpiparticles/)

A distributed marker (particle) container for MPI codes: per-rank
storage in component-major arrays on NumPy or CuPy (through
[cunumpy](https://github.com/max-models/cunumpy)), globally unique int64
ids, and `migrate`, which sends the markers that left a rank to the rank
that owns them. Without an MPI launcher it runs on one rank and makes no
MPI call.

``` python
import cunumpy as xp
import maybempi

from mpiparticles import (
    BoxDecomposition,
    IdAllocator,
    ParticleStore,
    global_count,
    migrate,
)

comm = maybempi.get_mpi().COMM_WORLD
decomposition = BoxDecomposition(comm, [0.0, 0.0], [1.0, 1.0], proc_sizes=[1, 1])
store = ParticleStore(ndim=2, nvel=3, id_allocator=IdAllocator.from_comm(comm))

store.add(xp.asarray([[0.2, 0.7], [0.4, 0.9]]), xp.zeros((3, 2)))
store.positions[:, : store.length] += 0.1  # push
sent, received = migrate(store, decomposition)  # collective: call it on every rank
store.compact_if_needed()
print(global_count(store, comm))
```

``` bash
mpiexec -n 4 python example.py   # with proc_sizes=[2, 2]; or just: python example.py
```

- **`ParticleStore`** holds positions `(ndim, n)`, velocities
  `(nvel, n)`, weights, alive flags and ids, with growth by doubling,
  dead markers until `compact`, `sort_by_key`, `snapshot`/`restore`, and
  a transposed non-copy view for plasmacoll’s `MarkerSet`.
- **`IdAllocator`** hands out int64 ids with a rank stride, independent
  of the decomposition.
- **`Decomposition`** is a protocol (`comm`, `rank`, `size`, `owners`,
  `owned_mask`); `BoxDecomposition` is a small reference implementation.
- **`migrate`** sends only the leavers, ids as int64 (exact above 2^53),
  and appends the arrivals. `keep_owned`, `global_count` and
  `gather_columns` are the companions.
- **`velocity_moments`** sums the velocity moments of every order over
  the markers of all ranks (`moment_indices` gives their order,
  `moment_values` the per-marker products for a deposit on a grid), and
  **`phase_space_density`** bins the markers on one to three phase-space
  axes (`BinAxis`), summed over the ranks.

Documentation: <https://max-models.github.io/mpiparticles/>

## Install

With [uv](https://docs.astral.sh/uv/):

``` bash
make install    # uv sync --extra dev, plus the pre-commit hooks
```

Or create and activate a Python environment (3.10 or newer) and install
with pip:

``` bash
pip install -e .          # serial
pip install -e ".[mpi]"   # with mpi4py, for mpiexec runs
```

The `test`, `docs` and `dev` extras install the test runner, the
documentation tooling and the linters:

``` bash
pip install -e ".[dev]"
```

## Development

Formatting and linting use [ruff](https://docs.astral.sh/ruff/), type
checking [pyright](https://microsoft.github.io/pyright/), run by
[pre-commit](https://pre-commit.com/) and in CI:

``` bash
make lint     # ruff check, ruff format --check, pyright
make test     # pytest with coverage; includes mpiexec -n 2 and -n 4 jobs when mpiexec is found
```

Commit messages follow [Conventional
Commits](https://www.conventionalcommits.org/); see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Build docs

The documentation in `docs/` is an Astro + Starlight site: hand-written
pages, the notebooks in `tutorials/` executed and published as pages,
complete Python scripts in `examples/` published with their plots, and
the API reference generated from the docstrings with
[starlight-pydocs](https://ewels.github.io/starlight-pydocs/). It needs
Node 22 or newer.

``` bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-examples    # run examples/*.py and publish their plots
make docs-dev         # live preview at http://localhost:4321/mpiparticles/
make docs-build       # the static site in docs/dist
```

## Build the README

`README.md` is rendered from `README.qmd` with
[Quarto](https://quarto.org/):

``` bash
make readme
```

## Releases

Before merging a release to `main`, update the version in
`pyproject.toml`, `src/mpiparticles/__init__.py` and `CITATION.cff`
(including its release date), and add the release notes to
`CHANGELOG.md`. The push to `main` creates a GitHub release with a
`vX.Y.Z` tag and publishes the package to PyPI with trusted publishing
(OIDC). The one-time PyPI and GitHub configuration is described in the
[publishing
tutorial](https://max-models.github.io/mpiparticles/development/publishing/).
