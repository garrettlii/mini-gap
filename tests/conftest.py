"""Phase gating.

  uv run pytest              phase 0 only (environment); should pass right after setup
  uv run pytest --phase 1    also phase-1 contract tests (fail until you implement them)
  uv run pytest --runslow    include 100-seed sweeps
Never edit a test to make it pass. Change the code, or log why the contract changed in
docs/DECISIONS.md and update the test in the same commit.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--phase", type=int, default=0, help="run tests up to this build phase")
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_collection_modifyitems(config, items):
    max_phase = config.getoption("--phase")
    runslow = config.getoption("--runslow")
    for item in items:
        m = item.get_closest_marker("phase")
        if m and m.args[0] > max_phase:
            item.add_marker(pytest.mark.skip(reason=f"phase {m.args[0]} (run with --phase {m.args[0]})"))
        if item.get_closest_marker("slow") and not runslow:
            item.add_marker(pytest.mark.skip(reason="slow (run with --runslow)"))


@pytest.fixture(scope="session")
def model():
    from gap.sim.model_builder import build_model
    return build_model()
