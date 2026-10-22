"""Quality-check plots: show a gel with lanes drawn on top.

Lanes are identified by NUMBER (1 = leftmost), with one colour per source:
  blue boxes      = Image Lab reference (hand-selected by the user)
  orange dashes   = lanes our detector found (straight from the comb)
  orange lines    = traced lanes (following the gel), red dots = bands used as evidence
  green rectangle = gel area our border detection found
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from gelanalyzer.border import GelArea
from gelanalyzer.imagelab import ImageLabAnalysis
from gelanalyzer.io import GelImage
from gelanalyzer.tracing import Lanes

REFERENCE_COLOR = "#1f6fd1"
DETECTED_COLOR = "#f28e2b"
AREA_COLOR = "#2ca02c"
EVIDENCE_COLOR = "#d62728"


def plot_lanes(
    gel: GelImage,
    detected: list[float] | None = None,
    reference: ImageLabAnalysis | None = None,
    area: GelArea | None = None,
    traced: Lanes | None = None,
    ax: Axes | None = None,
    contrast: tuple[float, float] = (1, 99.5),
) -> Axes:
    """Draw the gel (protein dark, like Image Lab) with reference and detected lanes.

    detected: x-positions (px) of lanes found by our detector.
    reference: Image Lab's saved analysis, see gelanalyzer.imagelab.
    area: gel area from gelanalyzer.border.find_gel_area.
    traced: lane paths from gelanalyzer.tracing.trace_lanes.
    contrast: lower/upper percentiles of pixel values mapped to white/black.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 9))

    vmin, vmax = np.percentile(gel.pixels, contrast)
    ax.imshow(gel.pixels, cmap="gray_r", vmin=vmin, vmax=vmax)
    height = gel.pixels.shape[0]
    legend = []

    if area is not None:
        ax.add_patch(Rectangle((area.left, area.top), area.width, area.height, fill=False,
                               edgecolor=AREA_COLOR, linewidth=1.5, linestyle="-.", gid="gel-area"))
        legend.append(Line2D([], [], color=AREA_COLOR, linestyle="-.", label="gel area (detected)"))

    if reference is not None:
        _draw_reference(ax, reference)
        legend.append(Line2D([], [], color=REFERENCE_COLOR, label="Image Lab lanes (reference)"))

    if detected is not None:
        for number, x in enumerate(sorted(detected), start=1):
            ax.axvline(x, color=DETECTED_COLOR, linestyle="--", linewidth=1.2, gid=f"detected-lane-{number}")
            ax.text(x, height - 5, str(number), color=DETECTED_COLOR, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", gid=f"detected-number-{number}")
        legend.append(Line2D([], [], color=DETECTED_COLOR, linestyle="--",
                             label=f"detected lanes ({len(detected)})"))

    if traced is not None:
        _draw_traced(ax, traced, height)
        legend.append(Line2D([], [], color=DETECTED_COLOR, label=f"traced lanes ({len(traced)})"))
        if traced.points:
            legend.append(Line2D([], [], color=EVIDENCE_COLOR, marker=".", linestyle="",
                                 label=f"bands used ({len(traced.points)})"))

    if legend:
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=len(legend), fontsize=9, frameon=False)
    title = gel.source.name if gel.source else "gel"
    if app := gel.metadata.get("application"):
        title += f"  ·  {app}"
    ax.set_title(title)
    return ax


def _draw_reference(ax: Axes, reference: ImageLabAnalysis) -> None:
    for lane in reference.lanes:
        half = lane.width / 2
        corners = [
            (lane.x_top - half, lane.y_top), (lane.x_top + half, lane.y_top),
            (lane.x_bottom + half, lane.y_bottom), (lane.x_bottom - half, lane.y_bottom),
        ]
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=REFERENCE_COLOR,
                             linewidth=2 if lane.is_standard else 1, gid=f"reference-lane-{lane.number}"))
        label = f"{lane.number} (L)" if lane.is_standard else str(lane.number)
        ax.text(lane.x_top, lane.y_top - 8, label, color=REFERENCE_COLOR, ha="center", va="bottom",
                fontsize=10, fontweight="bold", gid=f"reference-number-{lane.number}")
        for i, y in enumerate(lane.band_ys):
            x = lane.x_at(y)
            ax.plot([x - half, x + half], [y, y], color=REFERENCE_COLOR, linewidth=1.5,
                    gid=f"reference-band-{lane.number}-{i}")


def _draw_traced(ax: Axes, lanes: Lanes, height: float) -> None:
    ys = np.linspace(lanes.y_start, lanes.y_end, 60)
    paths = lanes.x_at(ys)
    for k in range(len(lanes)):
        ax.plot(paths[:, k], ys, color=DETECTED_COLOR, linewidth=1.3, gid=f"traced-lane-{k + 1}")
        for end, y in (("top", lanes.y_start), ("bottom", lanes.y_end)):     # broadness indicator
            x, half = lanes.x_at(y)[k], 0.85 * lanes.width_at(y)[k] / 2       # 85%: bars stay apart
            ax.plot([x - half, x + half], [y, y], color=DETECTED_COLOR, linewidth=2.5, solid_capstyle="butt",
                    gid=f"traced-width-{k + 1}-{end}")
        ax.text(paths[-1, k], height - 5, str(k + 1), color=DETECTED_COLOR, ha="center", va="bottom",
                fontsize=10, fontweight="bold", gid=f"traced-number-{k + 1}")
    if lanes.points:
        ax.plot([p.x for p in lanes.points], [p.y for p in lanes.points], ".", color=EVIDENCE_COLOR,
                markersize=4, gid="band-points")


def plot_ladder(gel: GelImage, ladder, reference_ys=None, reference_xs=None, identification=None):
    """Diagnostic figure for the ladder lane: the crop and its mask, side by side,
    plus (if `identification` is given) the lane profile with each band's kDa.

    Overlays: trapezoid (lane broadness), centre line, ladder blobs (red o),
    rejected blobs (magenta x), lane end (green dashes), optional reference
    bands (blue +, e.g. from Image Lab).
    """
    x0, x1, y0, y1 = ladder.crop
    crop = gel.pixels[y0:y1, x0:x1]
    if identification is None:
        fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(7, 9), sharey=True)
    else:
        fig, (ax_img, ax_mask, ax_prof) = plt.subplots(1, 3, figsize=(11, 9), sharey=True,
                                                       gridspec_kw={"width_ratios": [1, 1, 1.3]})
    vmin, vmax = np.percentile(crop, (1, 99.5))
    extent = (x0, x1, y1, y0)
    ax_img.imshow(crop, cmap="gray_r", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
    if ladder.mask is not None:
        ax_mask.imshow(ladder.mask, cmap="gray_r", extent=extent, aspect="auto")
    for ax in (ax_img, ax_mask):
        ax.add_patch(Polygon(ladder.trapezoid(), closed=True, fill=False, edgecolor=DETECTED_COLOR,
                             linewidth=1.5, gid="ladder-trapezoid"))
        ys = np.array([ladder.y_start, ladder.y_end])
        ax.plot(ladder.x_at(ys), ys, color=DETECTED_COLOR, linewidth=1, gid="ladder-centre")
        ax.plot([b.x for b in ladder.bands], [b.y for b in ladder.bands], "o", mfc="none",
                color=EVIDENCE_COLOR, markersize=7, gid="ladder-bands")
        if ladder.rejected:
            ax.plot([b.x for b in ladder.rejected], [b.y for b in ladder.rejected], "x",
                    color="m", markersize=6, gid="ladder-rejected")
        ax.axhline(ladder.y_end, color=AREA_COLOR, linestyle="--", linewidth=1, gid="ladder-end")
        if reference_ys is not None:
            ax.plot(reference_xs, reference_ys, "+", color=REFERENCE_COLOR, markersize=10, mew=2,
                    gid="ladder-reference")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    if identification is not None:
        _draw_ladder_profile(ax_prof, identification, reference_ys)
    name = gel.source.name if gel.source else "gel"
    ax_img.set_title(f"{name}\nladder lane {ladder.lane}: {len(ladder.bands)} bands", fontsize=9)
    ax_mask.set_title(f"mask, {len(ladder.rejected)} blobs rejected", fontsize=9)
    fig.tight_layout()
    return fig


def _draw_ladder_profile(ax: Axes, ident, reference_ys=None) -> None:
    ax.plot(ident.profile, ident.ys, color="gray", linewidth=0.8, gid="ladder-profile", label="lane profile")
    ax.plot(ident.background, ident.ys, color=AREA_COLOR, linewidth=1, gid="ladder-background",
            label="rolling-disc background")
    for band in ident.bands:
        x = float(np.interp(band.y, ident.ys, ident.profile))
        ax.plot(x, band.y, "o", mfc="none", color=EVIDENCE_COLOR, markersize=7)
        label = f"{band.kda:g} kDa" if band.kda is not None else "?"
        ax.text(x, band.y, "  " + label, va="center", fontsize=8, gid="ladder-kda",
                fontweight="bold" if band.thick else "normal", color=EVIDENCE_COLOR)
    for y in ident.spillage:
        ax.plot(float(np.interp(y, ident.ys, ident.profile)), y, "x", color="m", markersize=8, gid="ladder-spill")
    if reference_ys is not None:
        for y in reference_ys:
            ax.axhline(y, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8)
    status = "complete" if ident.complete else f"INCOMPLETE ({len(ident.bands)})"
    check = "thick bands OK" if ident.thick_bands_ok else "thick bands NOT confirmed"
    ax.set_title(f"profile: {status}, {check}", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
