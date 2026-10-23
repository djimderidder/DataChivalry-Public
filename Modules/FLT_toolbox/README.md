# FLT_toolbox

Analysis of PicoQuant FluoTime data:

* **Fluorescence lifetime**: global multi-exponential (1–4) reconvolution fits of TCSPC
  decays. Lifetimes are shared across datasets; amplitudes, background and scatter are
  per dataset. All IRF and shift variants that were tried are available as options.
* **Steady-state anisotropy**: wavelength-resolved r(λ) and G(λ) from VV/VH/HH/HV
  emission spectra, with optional blank subtraction and smoothing.

> **Status: archived.** Written for a project that was stopped (because of the data,
> not the code). Shared to document what was tried and as a tested starting point
> for similar data. Not actively developed.

## Run

1. Install the requirements: `pip install numpy scipy pandas matplotlib lmfit`
   (or `pip install -e ".[dev]"` to also get pytest).
2. Open `runFluoTime.py` and fill in the **USER SETTINGS**: `analysis`
   (`"demo"`, `"lifetime"` or `"anisotropy"`), `fileFolder`, the file lists with
   their metadata (concentration, condition, ...) and the fit config. Then run it
   (Spyder: *Run file*, F5). Leave `analysis = "demo"` to run on simulated data first.
3. Afterwards `result`, `summary`, `spectra`, ... are in the Variable Explorer. Set
   `save_results = True` to also write the figures (PNG) and tables (CSV) to `output/`.

From a terminal the same settings are used; flags override them without editing the
file: `python runFluoTime.py --analysis lifetime --data D:/flt/run2 --save output --no-show`.

### Data formats

Whitespace-separated text, no header (FluoTime text export):

| file | columns |
|---|---|
| lifetime decay | `time_ns  decay  irf` |
| anisotropy sample | `wavelength  VV  VH  HH  HV` |
| anisotropy blank | `wavelength  VV  VH` (HH, HV optional) |

Other column orders: `lifetime_columns` in the run script, or `load_decay(..., time_col,
counts_col, irf_col)` / `load_spectrum(..., channels=...)`. The time axis must be uniform.

## Tests

`pip install pytest`, then `pytest`. All tests use simulated data with known
parameters, so no data is needed. Tests on measured decays are skipped when
`tests/data/real/` is empty (git-ignored); run only those with `pytest -m real_data`.

## Limits

* Weights assume Poisson counts. Exports normalised to a peak of 10000 are only
  approximately Poisson, so read χ²ᵣ as a relative measure.
* Closely spaced lifetimes (within roughly a factor 1.5), or a lifetime sitting at a
  bound, give poorly determined fractions. Check the correlations in
  `result.report()` and compare against looser bounds.
* Not implemented: fitting a sub-range of the decay, incomplete-decay (previous
  pulse) correction, maximum-likelihood fitting, error estimates on fractions, and
  time-resolved anisotropy (the old VV/VH script was dropped because it did not
  fit rotational correlation times).

---

## Layout

```
runFluoTime.py           run script (USER SETTINGS at the top)
configs/                 lifetime fit models and constraints (TOML)
src/fluotime/
    io.py                load .dat files -> Decay / Spectrum objects
    irf.py               IRF preprocessing and time shifting
    models.py            reconvolution model (pure numpy, no fitting)
    lifetime/
        config.py        FitConfig + ParamSpec: every option and constraint
        parameters.py    FitConfig -> lmfit Parameters
        fit.py           fit_global() and LifetimeResult
        results.py       fractions and mean lifetimes
    anisotropy.py        G factor, r(λ), blank subtraction, smoothing, window average
    plotting.py          figures (returns them; the run script shows or saves)
    synthetic.py         simulated decays/spectra for the tests and the demo
tests/                   pytest suite
```

## Lifetime fitting

### Model

For dataset *k*, with the IRF normalised to unit sum and bin width Δt:

```
y_k(t) = Δt · [IRF(t − s_k) ⊗ Σ_i A_ik · exp(−t/τ_i)]  +  b_k  +  S_k · IRF(t − s_k)
```

| symbol | parameter | scope |
|---|---|---|
| τ_i | `tau{i}` | shared by all datasets |
| A_ik | `A{i}_ds{k}` | per dataset, ≥ 0, counts/ns (Σ counts ≈ Σ A_i τ_i) |
| b_k | `bkg_ds{k}` | per dataset, constant background per bin |
| S_k | `scatter_ds{k}` | per dataset, IRF-shaped scatter (total counts) |
| s_k | `shift` or `shift_ds{k}` | depends on `shift_mode` |

The fit minimises weighted residuals with lmfit (`leastsq` by default).
`result.report()` prints the lmfit report with the `ds{k}` → file name mapping.

### Fit options

All options live in `FitConfig` (`src/fluotime/lifetime/config.py`). Set them in a
config file, or in `fit_overrides` in the run script.

| option | values | meaning |
|---|---|---|
| `irf_source` | `"reference"` (default), `"own"` | one IRF (from `datasets[reference_index]`) for all, or each dataset its own IRF |
| `shift_mode` | `"global"` (default), `"per_dataset"`, `"anchored"` | one shift for all; a free shift per dataset; or fit the reference dataset alone, then let every shift vary within ± `anchor_window` ns of that value |
| `irf_preprocessing` | `"mean_threshold"` (default), `"percentile_subtract"`, `"none"` | see below |
| `weighting` | `"neyman"` (default), `"pearson"`, `"none"` | 1/√max(data,1), 1/√max(model,1), or unweighted |
| `method` | any lmfit method | e.g. `"leastsq"`, `"least_squares"`, `"nelder"` |

IRF preprocessing, always followed by normalising to unit sum:

- `mean_threshold` (original behaviour): the original code fitted lmfit's
  `ConstantModel` to the whole IRF trace, which is the mean of all bins, peak
  included. Bins below that value are set to zero and the rest are kept unchanged.
  It acts as a noise threshold; nothing is subtracted.
- `percentile_subtract`: subtracts the `irf_percentile`-th percentile as baseline and
  clips at zero.
- `none`: only clips negatives.

### Config files

`configs/` holds examples:
- `biexp_af488.toml`: the original two-component constraints
- `monoexp_free.toml`
- `triexp_fixed_dye.toml`: one lifetime fixed with `vary = false`

Every parameter takes `value`, `min`, `max` and `vary`:

```toml
n_exp = 2
shift_mode = "anchored"
anchor_window = 0.05
amplitude_guess_factors = [1.0, 1.0]   # start A_i = f_i * total_counts / (n_exp * tau_i)

[[taus]]
value = 4.1
min = 3.7
max = 4.2

[[taus]]
value = 2.8
min = 0.05
max = 3.8

[shift]
value = 0.0
min = -1.0
max = 1.0
# [background] and [scatter] work the same way (applied per dataset)
```

Unknown keys or invalid option values raise an error, so a typo fails loudly.

### Results

`result.summary()` gives one row per dataset with these columns:
- the metadata from the run script
- `tau1..n` and `A1..n`
- fractional amplitudes `a{i}_frac` and fractional intensities `f{i}_frac` (A·τ)
- amplitude- and intensity-weighted mean lifetimes, `tau_amp` and `tau_int`
- `shift`, `background`, `scatter` and `chi2_red`

Per dataset, `result.curve(k)`, `result.weighted_residuals(k)` and
`result.irf_on_grid(k)` return the fitted curve, the residuals and the shifted IRF.

### From the original scripts

The original analysis consisted of separate scripts that differed only in the options above:

| old script | equivalent settings |
|---|---|
| `mainFluoLifeTimeFirstIRF.py`, `…v2_2.py` | `irf_source="reference"`, `shift_mode="global"` |
| `mainFluoLifeTimeOwnIRF.py` | `irf_source="own"`, `shift_mode="global"` |
| `mainFluoLifeTimeOwnIRF_indShift.py` | `irf_source="own"`, `shift_mode="per_dataset"`, shift bounds ±0.5 |
| `mainFluoLifeTimeFirstIRF_indShift.py` | `irf_source="reference"`, `shift_mode="anchored"`, `anchor_window=0.05` |

The only deliberate change is the amplitude starting values. The old scripts used
`total/(n_exp·100)` times hand-tuned factors (70, 50 or 0.01); the new default
`total/(n_exp·τ_i)` already has the right scale. The old factors can be reproduced
with `amplitude_guess_factors`.

## Steady-state anisotropy

```
G(λ) = I_HV / I_HH
r(λ) = (I_VV − G·I_VH) / (I_VV + 2G·I_VH)
```

Steps (settings in the anisotropy part of the USER SETTINGS):

1. Gaussian smoothing of the sample channels (`smooth_sigma` in data points; `None` = off).
2. Blank subtraction of `subtract_from` (default VV and VH). The blank is
   interpolated onto the sample wavelengths and is *not* smoothed unless
   `smooth_blank = True`.
3. G(λ) and r(λ).
4. Mean ± std of r and G within `target_wavelength ± half_width`.

The defaults reproduce the original script. Some notes on them:
- G is a ratio, so leaving HV/HH without blank subtraction barely changes it when
  the blank is small compared with the signal. Otherwise use
  `subtract_from = ("VV", "VH", "HH", "HV")` with a four-channel blank.
- Smoothing the sample but not the blank only adds the blank's noise to r; it
  does not bias the window average. `smooth_blank = True` is identical to
  subtracting first and smoothing afterwards.

## What the tests cover

- **Loading, IRF handling and the model** (`test_io`, `test_irf`, `test_models`):
  - delta IRF gives a plain exponential
  - total area equals Σ A·τ
  - a shift delays the peak
  - the `mean_threshold` background equals the original `ConstantModel` fit
- **Parameter recovery** (`test_lifetime_fit`):
  - simulated biexponential decays with known τ, fractions and shifts are fitted
    with every `irf_source` and `shift_mode`
  - constraints and fixed parameters are respected
  - a too-simple model is detected by χ²
- **Configs** (`test_lifetime_config`): validation, including the files in `configs/`.
- **Anisotropy** (`test_anisotropy`):
  - formulas
  - recovery of known r and G
  - inputs are not modified
  - the blank and smoothing options
- **Run script** (`test_main`): runs it the Spyder way (no arguments, checks the
  variables it leaves behind) and from a terminal with flags.
- **Measured data** (`test_real_data`, marker `real_data`): fits the files in
  `tests/data/real/` with all option combinations and checks the results are finite
  and within bounds.
