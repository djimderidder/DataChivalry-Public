"""The notebook runs top to bottom on simulated data (magics skipped, no clicks)."""

import json
import re
from pathlib import Path

import matplotlib
import pytest

NOTEBOOK = Path(__file__).parents[1] / "FLUOstar Fluorescence Assay Disaggregation.ipynb"


def test_notebook_has_no_outputs_or_local_paths():
    """Outputs and a filled-in fileFolder would publish real data; settings stay None."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        assert not cell.get("outputs"), "strip notebook outputs before committing"
        for line in "".join(cell["source"]).splitlines():
            if re.match(r"\s*file(Folder|Name)\s*=", line):
                assert re.match(r"\s*file(Folder|Name)\s*=\s*None\b", line), line


def test_notebook_runs_on_simulated_data(monkeypatch):
    pytest.importorskip("ipywidgets")
    matplotlib.use("Agg")
    monkeypatch.chdir(NOTEBOOK.parent)
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(line for line in "".join(cell["source"]).splitlines()
                           if not line.lstrip().startswith("%"))
        exec(compile(source, "notebook", "exec"), namespace)
    assert len(namespace["summary"]) == 5
