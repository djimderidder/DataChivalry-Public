"""The run scripts keep working the Spyder way (no arguments) and from a terminal."""

import runpy
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

HERE = Path(__file__).parents[1]
ENV = {"MPLBACKEND": "Agg", "PATH": ""}


@pytest.fixture(autouse=True)
def agg(monkeypatch):
    matplotlib.use("Agg")


@pytest.mark.parametrize("script,expected", [
    ("runTht.py", {"data", "prepared", "norm", "kinetics", "summary"}),
    ("runThtConfig.py", {"config", "results", "summary"}),
])
def test_run_file_like_spyder(monkeypatch, script, expected):
    monkeypatch.setattr(sys, "argv", [script])
    namespace = runpy.run_path(str(HERE / script), run_name="__main__")
    assert expected <= set(namespace)
    assert len(namespace["summary"]) == 5   # five simulated sample wells


def test_runtht_from_terminal(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(HERE / "runTht.py"), "--wells", "A9,C9", "--save", str(out),
                           "--no-show"], capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stderr
    assert list(pd.read_csv(out / "tht_summary.csv")["well"]) == ["A9", "C9"]
    assert (out / "tht_overview.png").exists()


def test_runthtconfig_from_terminal(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(HERE / "runThtConfig.py"), "--rows", "1,3", "--diagnostics", "1",
                           "--save", str(out), "--no-show"], capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stderr
    assert list(pd.read_csv(out / "tht_summary.csv")["row"]) == [1, 3]
    assert (out / "diagnostics_row1_B9.png").exists()
