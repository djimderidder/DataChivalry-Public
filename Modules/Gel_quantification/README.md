# GelAnalyzer

Automatic analysis of SDS-PAGE gel images (Bio-Rad Image Lab `.scn` or 16-bit `.tif`).

**What it does now:** finds the gel area, the comb wells and the lanes, traces the
lanes down the gel, finds the ladder lane and identifies its bands with their size
(Precision Plus Unstained, 250 – 10 kDa).

## Run

1. Install the requirements: `pip install numpy scipy scikit-image tifffile matplotlib`
2. Open `runGelAnalyzer.py`, fill in `fileFolder`, `fileName` and the `config`
(number of lanes, ladder lane), and run it.
Leave `fileName = None` to run the built-in mock gel first.

## Tests

`pip install pytest`, then `pytest` (add `--save-plots` to write diagnostic images to
`reports/test-plots/`). Tests on real gels are skipped when `tests/data/real/` is empty.

## Limits

* Tested on three ChemiDoc images (Coomassie and stain-free) with the comb visible.
* Lane tracing for non-ladder lanes is still rough; band detection, kDa estimation
and quantification of samples are not built yet.
* more extensive version in private repo.

