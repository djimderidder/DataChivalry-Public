"""The run scripts keep working the Spyder way (no arguments) and from a terminal."""

import builtins
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
def agg():
    matplotlib.use("Agg")


def test_visualize_like_spyder(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["atpaseVisualize.py"])
    ns = runpy.run_path(str(HERE / "atpaseVisualize.py"), run_name="__main__")
    assert {"config", "traces", "summary"} <= set(ns)
    assert len(ns["summary"]) == 5


def test_picker_like_spyder_answering_yes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["atpasePicker.py"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    ns = runpy.run_path(str(HERE / "atpasePicker.py"), run_name="__main__")
    picks = ns["picks"]
    assert list(picks["row"]) == [0, 1, 2, 3, 4]        # every mock row misses Endi
    assert (picks["starti"] == 0).all()                  # the automatic proposal


def test_picker_manual_branch_uses_the_window_picker(monkeypatch):
    """Answer 'n': the window picker opens (closes at once without a GUI) and the fit is redone."""
    monkeypatch.setattr(sys, "argv", ["atpasePicker.py", "--rows", "1"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    monkeypatch.setattr("matplotlib.pyplot.pause", lambda *a: None)
    ns = runpy.run_path(str(HERE / "atpasePicker.py"), run_name="__main__")
    assert len(ns["picks"]) == 1


def test_visualize_from_terminal(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(HERE / "atpaseVisualize.py"), "--rows", "0,2",
                           "--save", str(out), "--no-show"], capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stderr
    assert list(pd.read_csv(out / "atpase_summary.csv")["row"]) == [0, 2]
    assert (out / "atpase_rates.png").exists()


def test_picker_from_terminal_writes_a_config_copy(tmp_path):
    proc = subprocess.run([sys.executable, str(HERE / "atpasePicker.py"), "--accept-all", "--write-back",
                           "--output", str(tmp_path)], capture_output=True, text=True, env=ENV)
    assert proc.returncode == 0, proc.stderr
    updated = pd.read_excel(tmp_path / "mock ATPase config (picked).xlsx")
    assert updated["Rate"].notna().all()
