"""Compare our lanes (traced down the gel) with Image Lab's saved lanes, one PNG per gel.

Usage:
    python scripts/compare_lanes.py tests/data/real/*.scn
    python scripts/compare_lanes.py my_gel.scn --lanes 15 --out reports --show

--lanes is the number of lanes (comb wells). Until we have a gel config (US-4),
it falls back to the number of lanes in the Image Lab analysis, if there is one.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from gelanalyzer.border import find_gel_area
from gelanalyzer.imagelab import read_imagelab_analysis
from gelanalyzer.io import load_gel
from gelanalyzer.tracing import trace_lanes
from gelanalyzer.plotting import plot_lanes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help=".scn or .tif gel images")
    parser.add_argument("--lanes", type=int, help="number of lanes / comb wells")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="folder for the PNGs (default: reports)")
    parser.add_argument("--show", action="store_true", help="also open each plot in a window")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for path in args.files:
        gel = load_gel(path)
        reference = read_imagelab_analysis(path) if path.suffix.lower() == ".scn" else None
        n_lanes = args.lanes or (len(reference.lanes) if reference else None)
        if n_lanes is None:
            print(f"{path.name}: skipped, give the number of lanes with --lanes")
            continue

        area = find_gel_area(gel.pixels)
        lanes = trace_lanes(gel.pixels, n_lanes=n_lanes, area=area)
        ax = plot_lanes(gel, reference=reference, area=area, traced=lanes)
        out = args.out / f"{path.stem}_lanes.png"
        ax.figure.savefig(out, dpi=100, bbox_inches="tight")

        ref_text = f"Image Lab {len(reference.lanes)} lanes" if reference else "no Image Lab reference"
        print(f"{path.name}: traced {len(lanes)} lanes from {len(lanes.points)} bands, {ref_text}  ->  {out}")
        if args.show:
            plt.show()
        plt.close(ax.figure)


if __name__ == "__main__":
    main()
