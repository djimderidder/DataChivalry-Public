"""Loading gel images from disk.

Convention for the whole package: in `GelImage.pixels` a HIGH value means
MORE protein. Bio-Rad Image Lab already stores .scn files that way (for both
Coomassie and stain-free). For other sources, use `GelImage.inverted()`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import tifffile

from gelanalyzer.config import GelConfig


class ScnFormatError(ValueError):
    """The file looks like a .scn but can't be read as one."""


@dataclass
class GelImage:
    """A gel image plus everything we know about where it came from."""

    pixels: np.ndarray                                   # 2D float, (height, width), high = protein
    metadata: dict = field(default_factory=dict)         # e.g. application, imager, exposure_time_s
    source: Path | None = None                           # file it was loaded from
    pixel_size_mm: tuple[float, float] | None = None     # (x, y) mm per pixel, if known
    config: GelConfig | None = None                      # what the user told us (lanes, ladder, ...)

    def inverted(self) -> GelImage:
        """Return a copy with polarity flipped (dark bands <-> bright bands).

        Flips within the image's own range, so min and max swap places.
        """
        lo, hi = self.pixels.min(), self.pixels.max()
        return replace(self, pixels=hi + lo - self.pixels, metadata=dict(self.metadata))


def load_gel(path: str | Path, config: GelConfig | dict | None = None) -> GelImage:
    """Load a gel image. Supported: Bio-Rad Image Lab .scn, and .tif/.tiff.

    config: the user's gel config (a GelConfig or a plain dict), attached as `gel.config`.
    """
    path = Path(path)
    if isinstance(config, dict):
        config = GelConfig.from_dict(config)
    suffix = path.suffix.lower()
    if suffix == ".scn":
        return replace(_load_scn(path), config=config)
    if suffix in (".tif", ".tiff"):
        return replace(_load_tif(path), config=config)
    raise ValueError(f"Unsupported file type '{suffix}' ({path.name}). Use .scn or .tif.")


# --- TIFF ----------------------------------------------------------------------

def _load_tif(path: Path) -> GelImage:
    pixels = tifffile.imread(path)
    if pixels.ndim != 2:
        raise ValueError(f"{path.name}: expected a single grayscale image, got shape {pixels.shape}")
    return GelImage(pixels=pixels.astype(np.float64), source=path)


# --- Bio-Rad .scn ----------------------------------------------------------------
# A .scn is a MIME multipart container (see tests/scn_factory.py for a diagram).
# We walk through it header by header. Parts with a Content-Length have a body
# we read and then jump over -- that way we never search inside binary pixel data.

_HEADER_BLOCK = re.compile(rb"Content-Type:[^\r\n]*\r\n(?:[\w-]+:[^\r\n]*\r\n)*\r\n")

# scan_attributes tag -> (our metadata key, converter)
_KNOWN_ATTRIBUTES = {
    "application": ("application", str),
    "imager": ("imager", str),
    "exposure_time": ("exposure_time_s", float),
    "image_date": ("image_date", str),
    "excitation_source": ("excitation_source", str),
    "emission_filter": ("emission_filter", str),
}


def read_scn_parts(path: str | Path) -> dict[str, bytes]:
    """Return {Content-Description: body} for every part of a .scn that has a body.

    Typical keys: ItemHeaderTag, ImageData, ImageHeader, ItemProtocolSettingsTag,
    ItemLaneAndBandDataTag.
    """
    data = Path(path).read_bytes()
    parts: dict[str, bytes] = {}
    pos = 0
    while (match := _HEADER_BLOCK.search(data, pos)):
        headers = dict(
            line.split(": ", 1)
            for line in match.group().decode("latin-1").strip().split("\r\n")
        )
        pos = match.end()
        if "Content-Length" in headers:                 # a leaf part: read body, skip past it
            length = int(headers["Content-Length"])
            parts.setdefault(headers.get("Content-Description", ""), data[pos:pos + length])
            pos += length
        # else: a multipart container -- just continue into it
    return parts


def _load_scn(path: Path) -> GelImage:
    parts = read_scn_parts(path)
    for required in ("ImageData", "ImageHeader"):
        if required not in parts:
            raise ScnFormatError(f"{path.name}: no '{required}' part found -- is this an Image Lab .scn file?")

    header = ET.fromstring(parts["ImageHeader"])
    size = header.find("size_pix")
    width, height = int(size.get("width")), int(size.get("height"))

    raw = parts["ImageData"]
    if len(raw) != width * height * 2:
        raise ScnFormatError(
            f"{path.name}: image size {width}x{height} needs {width * height * 2} bytes, "
            f"but ImageData has {len(raw)}")
    pixels = np.frombuffer(raw, dtype="<u2").reshape(height, width).astype(np.float64)

    metadata = {}
    for attribute in header.iterfind("scan_attributes/*"):
        key, convert = _KNOWN_ATTRIBUTES.get(attribute.tag, (attribute.tag, str))
        metadata[key] = convert(attribute.get("value"))
    if (zero_is := header.find("image")) is not None:
        metadata["zero_is"] = zero_is.get("zero_is")

    pixel_size_mm = None
    mm = header.find("size_mm")
    if mm is not None and mm.get("known") == "true":
        pixel_size_mm = (float(mm.get("width")) / width, float(mm.get("height")) / height)

    return GelImage(pixels=pixels, metadata=metadata, source=path, pixel_size_mm=pixel_size_mm)
