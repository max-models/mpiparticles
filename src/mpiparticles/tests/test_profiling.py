"""mpiparticles' scope-profiler regions are opt-in."""

import pytest
import scope_profiler as sp

from mpiparticles import BoxDecomposition, migrate
from mpiparticles.tests.conftest import FakeComm, filled_store


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    """Keep the profiling state of one test out of the others."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _profiled_region_names(**setup_kwargs):
    """Return the regions recorded while migrating and compacting a store."""
    sp.setup(output=None, **setup_kwargs)
    store = filled_store(5, capacity=4)  # grows once
    migrate(store, BoxDecomposition(FakeComm(0, 1), [0, 0], [1, 1], [1, 1]))
    store.compact()
    with sp.region("user.step"):
        pass
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_regions_are_off_by_default():
    assert _profiled_region_names() == {"user.step"}


def test_enable_tags_turns_the_regions_on():
    names = _profiled_region_names(enable_tags=["mpiparticles"])
    assert names == {
        "mpiparticles.compact",
        "mpiparticles.migrate",
        "mpiparticles.reserve",
        "user.step",
    }


def test_the_opt_in_tag_has_an_entry_point():
    """``scope-profiler tags`` finds the tag through the package's entry point."""
    from importlib.metadata import entry_points

    from mpiparticles._profiling import PROFILE_TAGS

    found = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="scope_profiler.opt_in_tags")
    }
    assert found.get("mpiparticles") == "mpiparticles"
    assert "mpiparticles" in PROFILE_TAGS
