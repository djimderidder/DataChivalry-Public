# -*- coding: utf-8 -*-
"""
Analyse an SDS-PAGE gel image: gel area, lanes, ladder lane and ladder band sizes.

Fill in the USER SETTINGS below and run this file.
Leave fileName = None to run the built-in mock example (a synthetic gel), so
you can see what the analysis does without any data.

Supported images: Bio-Rad Image Lab .scn, and 16-bit .tif.
@author: ridderdde
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt

from gelanalyzer.border import find_gel_area
from gelanalyzer.comb import find_comb
from gelanalyzer.config import GelConfig
from gelanalyzer.io import GelImage, load_gel
from gelanalyzer.ladder import find_ladder, identify_ladder_bands
from gelanalyzer.plotting import plot_ladder, plot_lanes
from gelanalyzer.tracing import trace_lanes

# ---------- USER SETTINGS ----------
fileFolder = None        # folder with the gel image, e.g. r"D:\experiments\gels"
fileName = None          # e.g. "my_gel.scn"   (None = run the mock example)

config = {
    "n_lanes": 15,                          # number of wells in the comb
    "ladder_lanes": [4],                    # lane(s) with the ladder, 1 = leftmost
    "ladder": "Precision Plus Unstained",   # 250 ... 10 kDa, thick 75/50/25
    "gel_type": "4-20% gradient",
}

show_figures = True      # open the figures in a window
save_figures = False     # write the figures as PNG next to this script (output/)
# -------------------------------------------------


def mock_gel() -> GelImage:
    """A synthetic gel: comb, 10 lanes fanning out, a 10-band ladder in lane 4."""
    from gelanalyzer.synthetic import make_gel

    truth = make_gel(n_lanes=10, width=900, height=600, margin=(40, 30, 50, 35), comb=True,
                     bands_per_lane=4, fan=40, ladder_lane=3, seed=7)
    mock_config = GelConfig.from_dict({"n_lanes": 10, "ladder_lanes": [4], "ladder": "Precision Plus Unstained"})
    return GelImage(pixels=truth.image, config=mock_config, metadata={"application": "mock gel"})


# 1 - load the gel and the config
if fileName is None:
    print("No fileName given: running the mock example (synthetic gel, 10 lanes, ladder in lane 4).")
    gel = mock_gel()
else:
    gel = load_gel(Path(fileFolder) / fileName, config=config)
n_lanes = gel.config.n_lanes

# 2 - where is the gel, where are the lanes
area = find_gel_area(gel.pixels)
comb = find_comb(gel.pixels, n_lanes=n_lanes, area=area)
lanes = trace_lanes(gel.pixels, n_lanes=n_lanes, area=area, comb=comb)

# 3 - the ladder lane and its band sizes
ladder = find_ladder(gel, area=area, comb=comb)
ladder_bands = identify_ladder_bands(gel, ladder)

print(f"Gel area: x {area.left:.0f}-{area.right:.0f}, y {area.top:.0f}-{area.bottom:.0f} px")
print(f"Lanes: {n_lanes}, spacing {comb.pitch:.0f} px, lanes start at y = {comb.bottom:.0f} px")
print(f"Ladder lane {ladder.lane}: {len(ladder_bands.bands)} bands "
      f"({'complete' if ladder_bands.complete else 'INCOMPLETE'}, "
      f"thick bands {'OK' if ladder_bands.thick_bands_ok else 'not confirmed'})")
for band in ladder_bands.bands:
    size = f"{band.kda:g} kDa" if band.kda is not None else "?"
    print(f"   {size:>8}   y = {band.y:6.0f} px{'   (thick)' if band.thick else ''}")

# 4 - figures
ax = plot_lanes(gel, area=area, traced=lanes)
fig_ladder = plot_ladder(gel, ladder, identification=ladder_bands)
if save_figures:
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    stem = Path(fileName).stem if fileName else "mock_gel"
    ax.figure.savefig(out / f"{stem}_lanes.png", dpi=100, bbox_inches="tight")
    fig_ladder.savefig(out / f"{stem}_ladder.png", dpi=100, bbox_inches="tight")
    print(f"Figures saved in {out}")
if show_figures:
    plt.show()
