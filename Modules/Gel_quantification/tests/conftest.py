"""Shared test fixtures. pytest loads this file automatically.

Diagnostic images: run  pytest --save-plots  to let tests write their
diagnostic pictures to reports/test-plots/ (off by default, so normal runs
stay fast and write nothing).
"""

from pathlib import Path

import pytest

from synthetic import make_gel

PLOT_DIR = Path(__file__).parent.parent / "reports" / "test-plots"


def pytest_addoption(parser):
    parser.addoption("--save-plots", action="store_true", default=False,
                     help="write diagnostic images from tests to reports/test-plots/")


@pytest.fixture
def save_plot(request):
    """Call save_plot(name, figure). Writes a PNG only when pytest runs with --save-plots."""
    import matplotlib.pyplot as plt

    enabled = request.config.getoption("--save-plots")

    def save(name: str, figure) -> Path | None:
        path = None
        if enabled:
            PLOT_DIR.mkdir(parents=True, exist_ok=True)
            path = PLOT_DIR / f"{name}.png"
            figure.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(figure)
        return path

    return save


@pytest.fixture
def simple_gel():
    """A clean, easy gel: 6 lanes, little noise."""
    return make_gel(n_lanes=6, noise=0.02, seed=1)


@pytest.fixture
def noisy_gel():
    """Harder: more lanes and a lot more noise."""
    return make_gel(n_lanes=10, width=900, noise=0.08, seed=2)
