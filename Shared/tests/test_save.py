import pandas as pd

from fluostar.io import load_fluostar
from fluostar.save import append_plate_to_csv


def test_append_creates_then_appends(three_gain_csv, tmp_path):
    data = load_fluostar(three_gain_csv)
    roi = pd.DataFrame({"well": ["A9", "B9"], "nFibril": ["H10"] * 2, "nBase": ["H11"] * 2})
    master = tmp_path / "out" / "all_wells.csv"

    append_plate_to_csv(data, roi, master)
    append_plate_to_csv(data, roi.iloc[:1], master)

    df = pd.read_csv(master)
    assert len(df) == 3
    assert list(df.columns) == ["fileFolder", "fileName", "well", "condition", "nFibril", "nBase", "date"]
    assert df["condition"].iloc[0] == "content A9;"


def test_save_and_show_writes_files(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fluostar.save import save_and_show

    fig = plt.figure()
    save_and_show({"fig": fig}, {"summary": pd.DataFrame({"a": [1]})}, tmp_path / "out", show=False)
    assert (tmp_path / "out" / "fig.png").exists()
    assert pd.read_csv(tmp_path / "out" / "summary.csv")["a"].iloc[0] == 1
    assert not plt.fignum_exists(fig.number)
