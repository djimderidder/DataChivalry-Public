"""Smoke tests: runFluoTime.py keeps working both ways it is meant to be run."""

import runpy
import subprocess
import sys
from pathlib import Path

import matplotlib

MAIN = Path(__file__).parents[1] / "runFluoTime.py"


def test_run_file_like_spyder(monkeypatch):
    """Spyder's "Run file" ~ runpy with no arguments: the USER SETTINGS are used and the
    results end up as variables in the namespace (Spyder's Variable Explorer)."""
    matplotlib.use("Agg")  # no windows during tests; plt.show() becomes a no-op
    monkeypatch.setattr(sys, "argv", [str(MAIN)])

    namespace = runpy.run_path(str(MAIN), run_name="__main__")

    assert namespace["analysis"] == "demo"
    assert len(namespace["summary"]) == 4
    assert {"result", "spectra", "anisotropy_summary"} <= set(namespace)


def test_run_from_terminal_with_flags(tmp_path):
    out = tmp_path / "results"
    proc = subprocess.run(
        [sys.executable, str(MAIN), "--analysis", "demo", "--save", str(out), "--no-show"],
        capture_output=True, text=True, env={"MPLBACKEND": "Agg", "PATH": ""},
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "lifetime_summary.csv").exists()
    assert (out / "anisotropy_summary.csv").exists()
    assert len(list(out.glob("*.png"))) == 12


def test_missing_data_gives_a_clear_error(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(MAIN), "--analysis", "lifetime", "--data", str(tmp_path), "--no-show"],
        capture_output=True, text=True, env={"MPLBACKEND": "Agg", "PATH": ""},
    )
    assert proc.returncode != 0
    assert "Check `fileFolder`" in proc.stderr
