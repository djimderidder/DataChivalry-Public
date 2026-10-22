"""US-2: load gel images. Written before the code (TDD)."""

import numpy as np
import pytest
import tifffile

from gelanalyzer.io import GelImage, ScnFormatError, load_gel
from scn_factory import write_fake_scn


@pytest.fixture
def pixels():
    """A small 16-bit image with a recognisable pattern (not square, so we catch w/h mix-ups)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 65535, size=(30, 50), dtype=np.uint16)


# --- .scn ------------------------------------------------------------------

def test_scn_pixels_are_loaded_exactly(tmp_path, pixels):
    gel = load_gel(write_fake_scn(tmp_path / "gel.scn", pixels))
    assert isinstance(gel, GelImage)
    assert gel.pixels.shape == (30, 50)          # (height, width)
    np.testing.assert_array_equal(gel.pixels, pixels)


def test_pixels_are_float_so_maths_cannot_overflow(tmp_path, pixels):
    # uint16 silently wraps around: np.uint16(10) - np.uint16(20) == 65526
    gel = load_gel(write_fake_scn(tmp_path / "gel.scn", pixels))
    assert gel.pixels.dtype == np.float64


def test_scn_metadata(tmp_path, pixels):
    gel = load_gel(write_fake_scn(tmp_path / "gel.scn", pixels, application="Stain Free"))
    assert gel.metadata["application"] == "Stain Free"
    assert gel.metadata["imager"] == "ChemiDoc MP"
    assert gel.metadata["exposure_time_s"] == pytest.approx(0.49)


def test_scn_pixel_size_in_mm(tmp_path, pixels):
    # 50 px wide over 10 mm -> 0.2 mm per pixel
    gel = load_gel(write_fake_scn(tmp_path / "gel.scn", pixels, size_mm=(10.0, 6.0)))
    assert gel.pixel_size_mm == pytest.approx((0.2, 0.2))


def test_scn_without_image_data_gives_clear_error(tmp_path, pixels):
    path = write_fake_scn(tmp_path / "broken.scn", pixels, include_image=False)
    with pytest.raises(ScnFormatError, match="ImageData"):
        load_gel(path)


def test_scn_with_wrong_number_of_bytes_gives_clear_error(tmp_path, pixels):
    # header claims a bigger image than the data contains
    path = write_fake_scn(tmp_path / "gel.scn", pixels)
    path.write_bytes(path.read_bytes().replace(b'height="30"', b'height="31"'))
    with pytest.raises(ScnFormatError, match="size"):
        load_gel(path)


# --- .tif ------------------------------------------------------------------

def test_tif_16bit_is_loaded(tmp_path, pixels):
    path = tmp_path / "gel.tif"
    tifffile.imwrite(path, pixels)
    gel = load_gel(path)
    np.testing.assert_array_equal(gel.pixels, pixels)
    assert gel.pixels.dtype == np.float64


# --- general ---------------------------------------------------------------

def test_unsupported_extension(tmp_path):
    path = tmp_path / "gel.jpg"
    path.write_bytes(b"not really")
    with pytest.raises(ValueError, match=r"\.jpg"):
        load_gel(path)


def test_source_path_is_remembered(tmp_path, pixels):
    path = write_fake_scn(tmp_path / "gel.scn", pixels)
    assert load_gel(path).source == path


# --- US-3: polarity --------------------------------------------------------
# Convention for the whole package: HIGH value = MORE protein.
# Image Lab .scn already stores it that way; other files may need flipping.

def test_inverted_flips_polarity_within_the_data_range():
    gel = GelImage(pixels=np.array([[0.0, 10.0, 40.0]]))
    flipped = gel.inverted()
    np.testing.assert_array_equal(flipped.pixels, [[40.0, 30.0, 0.0]])


def test_inverted_returns_a_new_gel_and_keeps_the_original():
    gel = GelImage(pixels=np.array([[1.0, 2.0]]), metadata={"application": "x"})
    flipped = gel.inverted()
    np.testing.assert_array_equal(gel.pixels, [[1.0, 2.0]])   # untouched
    assert flipped.metadata == gel.metadata
