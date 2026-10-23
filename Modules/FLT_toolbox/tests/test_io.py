import numpy as np
import pytest

from fluotime.io import Decay, Spectrum, load_decay, load_spectrum


def _write(path, rows):
    path.write_text("\n".join(" ".join(f"{v:g}" for v in row) for row in rows) + "\n")


# --- decays -----------------------------------------------------------------

def test_load_decay_three_columns(tmp_path):
    f = tmp_path / "sample.dat"
    _write(f, [(0.0, 5, 0), (0.032, 7, 1), (0.064, 9, 3), (0.096, 4, 2)])

    d = load_decay(f)

    assert isinstance(d, Decay)
    np.testing.assert_allclose(d.time, [0, 0.032, 0.064, 0.096])
    np.testing.assert_allclose(d.counts, [5, 7, 9, 4])
    np.testing.assert_allclose(d.irf, [0, 1, 3, 2])
    assert d.dt == pytest.approx(0.032)
    assert d.name == "sample"  # defaults to the file stem


def test_load_decay_name_and_metadata(tmp_path):
    f = tmp_path / "x.dat"
    _write(f, [(0, 1, 0), (1, 2, 0), (2, 3, 0)])

    d = load_decay(f, name="0 uM", metadata={"concentration": 0.0})

    assert d.name == "0 uM"
    assert d.metadata == {"concentration": 0.0}


def test_load_decay_custom_columns_and_no_irf(tmp_path):
    f = tmp_path / "x.dat"
    _write(f, [(9, 0.0, 1), (9, 0.1, 2), (9, 0.2, 3)])

    d = load_decay(f, time_col=1, counts_col=2, irf_col=None)

    np.testing.assert_allclose(d.time, [0, 0.1, 0.2])
    np.testing.assert_allclose(d.counts, [1, 2, 3])
    assert d.irf is None


def test_decay_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        Decay(time=np.arange(3.0), counts=np.ones(4), irf=None)


def test_decay_rejects_non_uniform_time():
    with pytest.raises(ValueError, match="uniform"):
        Decay(time=np.array([0, 1, 3, 4.0]), counts=np.ones(4), irf=None)


def test_load_decay_too_few_columns(tmp_path):
    f = tmp_path / "x.dat"
    _write(f, [(0, 1), (1, 2), (2, 3)])
    with pytest.raises(ValueError, match="column"):
        load_decay(f)  # default expects an IRF in column 2


# --- spectra ----------------------------------------------------------------

def test_load_spectrum_five_columns(tmp_path):
    f = tmp_path / "ani.dat"
    _write(f, [(500, 10, 5, 8, 7), (501, 11, 6, 9, 8)])

    s = load_spectrum(f)

    assert isinstance(s, Spectrum)
    np.testing.assert_allclose(s.wavelength, [500, 501])
    np.testing.assert_allclose(s["VV"], [10, 11])
    np.testing.assert_allclose(s["VH"], [5, 6])
    np.testing.assert_allclose(s["HH"], [8, 9])
    np.testing.assert_allclose(s["HV"], [7, 8])


def test_load_blank_with_two_channels(tmp_path):
    f = tmp_path / "blank.dat"
    _write(f, [(500, 1, 2), (501, 3, 4)])

    b = load_spectrum(f, channels=("VV", "VH"))

    assert set(b.channels) == {"VV", "VH"}
    np.testing.assert_allclose(b["VH"], [2, 4])


def test_spectrum_to_frame(tmp_path):
    f = tmp_path / "ani.dat"
    _write(f, [(500, 10, 5, 8, 7), (501, 11, 6, 9, 8)])
    df = load_spectrum(f).to_frame()
    assert list(df.columns) == ["wavelength", "VV", "VH", "HH", "HV"]
