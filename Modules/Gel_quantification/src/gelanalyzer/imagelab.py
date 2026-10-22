"""Read the lane/band analysis that Bio-Rad Image Lab saved inside a .scn file.

FOR VALIDATION ONLY. These are lanes and bands a person selected by hand in
Image Lab (often only the bands of interest), so they are a reference to
compare our own detection against -- not an input to the analysis pipeline.

How Image Lab stores geometry (worked out from real files):
  * a rectangular lane *frame* (left, top, right, bottom) in pixels
  * per lane: top_anchor / bottom_anchor = lane centre at the frame's top /
    bottom edge, as a fraction of the frame width -> lanes are slanted lines
  * per band: rel_peak = position along the lane, as a fraction of frame height
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from gelanalyzer.io import read_scn_parts


@dataclass
class ReferenceLane:
    number: int                     # 1-based, left to right, as in Image Lab
    label: str
    x_top: float                    # lane centre at the top of the frame (px)
    x_bottom: float                 # lane centre at the bottom of the frame (px)
    y_top: float
    y_bottom: float
    width: float                    # px
    is_standard: bool               # Image Lab "std_lane": the ladder
    band_ys: list[float] = field(default_factory=list)   # band peak positions (px)

    def x_at(self, y: float) -> float:
        """Lane centre at height y (lanes can be slanted)."""
        t = (y - self.y_top) / (self.y_bottom - self.y_top)
        return self.x_top + t * (self.x_bottom - self.x_top)

    @property
    def x_center(self) -> float:
        """Lane centre halfway down the frame."""
        return (self.x_top + self.x_bottom) / 2


@dataclass
class ImageLabAnalysis:
    frame: tuple[float, float, float, float]       # left, top, right, bottom (px)
    lanes: list[ReferenceLane]
    standard_name: str | None = None
    standard_kda: list[float] = field(default_factory=list)

    @property
    def standard_lane(self) -> ReferenceLane | None:
        return next((lane for lane in self.lanes if lane.is_standard), None)


def read_imagelab_analysis(path: str | Path) -> ImageLabAnalysis | None:
    """Return Image Lab's saved analysis, or None if the file has none."""
    xml = read_scn_parts(path).get("ItemLaneAndBandDataTag")
    if xml is None:
        return None
    frame_el = ET.fromstring(xml).find(".//lane_frame")
    if frame_el is None:
        return None

    def point(tag):
        el = frame_el.find(tag)
        return float(el.get("x")), float(el.get("y"))

    (left, top), (right, _) = point("top_left"), point("top_right")
    _, bottom = point("bottom_left")
    frame_w, frame_h = right - left, bottom - top

    lanes = []
    for number, lane in enumerate(frame_el.iterfind("lanes/lane"), start=1):
        lanes.append(ReferenceLane(
            number=number,
            label=(lane.findtext("label") or "").strip(),
            x_top=left + float(lane.findtext("top_anchor")) * frame_w,
            x_bottom=left + float(lane.findtext("bottom_anchor")) * frame_w,
            y_top=top,
            y_bottom=bottom,
            width=float(lane.findtext("lane_width")),
            is_standard=lane.findtext("std_lane") == "true",
            band_ys=[top + float(b.get("rel_peak")) * frame_h
                     for b in lane.iterfind("bands/band")],
        ))

    return ImageLabAnalysis(
        frame=(left, top, right, bottom),
        lanes=lanes,
        standard_name=frame_el.findtext("standard/name"),
        standard_kda=[float(v.get("value")) for v in frame_el.iterfind("standard/std_values/std_value")],
    )
