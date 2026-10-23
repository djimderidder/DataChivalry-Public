"""Global reconvolution fit: shared lifetimes, per-dataset amplitudes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from lmfit import Minimizer, Parameters, fit_report
from lmfit.minimizer import MinimizerResult

from ..io import Decay
from ..irf import preprocess_irf, shift_irf
from ..models import reconvolution_model
from .config import FitConfig
from .parameters import build_parameters, dataset_labels
from .results import fractions, mean_lifetime

IRF = tuple[np.ndarray, np.ndarray]  # (time axis of the IRF, normalised IRF)


# --- building blocks ------------------------------------------------------------

def prepare_irfs(datasets: Sequence[Decay], config: FitConfig) -> list[IRF]:
    """Pick and preprocess the IRF for every dataset according to ``config.irf_source``."""
    def clean(decay: Decay) -> IRF:
        if decay.irf is None:
            raise ValueError(f"dataset {decay.name!r} has no IRF column")
        return decay.time, preprocess_irf(decay.irf, config.irf_preprocessing, config.irf_percentile)

    if config.irf_source == "own":
        return [clean(d) for d in datasets]

    if not 0 <= config.reference_index < len(datasets):
        raise ValueError(f"reference_index={config.reference_index} but only "
                         f"{len(datasets)} dataset(s) given")
    reference = clean(datasets[config.reference_index])
    return [reference] * len(datasets)


def compute_weights(data: np.ndarray, model: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "neyman":
        return 1.0 / np.sqrt(np.clip(data, 1.0, None))
    if scheme == "pearson":
        return 1.0 / np.sqrt(np.clip(model, 1.0, None))
    if scheme == "none":
        return np.ones_like(data)
    raise ValueError(f"unknown weighting {scheme!r}")


def _shift_name(label: str, config: FitConfig) -> str:
    return "shift" if config.shift_mode == "global" else f"shift_{label}"


def model_curve(params: Parameters, index: int, datasets: Sequence[Decay],
                irfs: Sequence[IRF], config: FitConfig) -> np.ndarray:
    label = f"ds{index}"
    p = params.valuesdict()
    t_irf, irf = irfs[index]
    return reconvolution_model(
        datasets[index].time, irf,
        amplitudes=[p[f"A{i}_{label}"] for i in range(1, config.n_exp + 1)],
        taus=[p[f"tau{i}"] for i in range(1, config.n_exp + 1)],
        shift=p[_shift_name(label, config)],
        background=p[f"bkg_{label}"],
        scatter=p[f"scatter_{label}"],
        t_irf=t_irf,
    )


def weighted_residuals(params: Parameters, index: int, datasets: Sequence[Decay],
                       irfs: Sequence[IRF], config: FitConfig) -> np.ndarray:
    data = datasets[index].counts
    model = model_curve(params, index, datasets, irfs, config)
    return (data - model) * compute_weights(data, model, config.weighting)


def objective(params: Parameters, datasets: Sequence[Decay], irfs: Sequence[IRF],
              config: FitConfig) -> np.ndarray:
    return np.concatenate([weighted_residuals(params, k, datasets, irfs, config)
                           for k in range(len(datasets))])


def estimate_reference_shift(datasets: Sequence[Decay], irfs: Sequence[IRF],
                             config: FitConfig) -> float:
    """Fit only the reference dataset (global shift) and return its IRF shift."""
    k = config.reference_index
    if not 0 <= k < len(datasets):
        raise ValueError(f"reference_index={k} out of range")
    single = config.with_overrides(shift_mode="global", reference_index=0)
    params = build_parameters([datasets[k]], single)
    res = Minimizer(objective, params, fcn_args=([datasets[k]], [irfs[k]], single)).minimize(
        method=config.method)
    return float(res.params["shift"].value)


# --- result object ----------------------------------------------------------------

@dataclass
class LifetimeResult:
    datasets: list[Decay]
    config: FitConfig
    irfs: list[IRF]
    fit: MinimizerResult
    shift_reference: float | None = None

    @property
    def labels(self) -> list[str]:
        return dataset_labels(len(self.datasets))

    @property
    def params(self) -> Parameters:
        return self.fit.params

    @property
    def success(self) -> bool:
        return bool(self.fit.success)

    @property
    def taus(self) -> np.ndarray:
        return np.array([self.params[f"tau{i}"].value for i in range(1, self.config.n_exp + 1)])

    def amplitudes(self, index: int) -> np.ndarray:
        return np.array([self.params[f"A{i}_ds{index}"].value
                         for i in range(1, self.config.n_exp + 1)])

    def shift(self, index: int) -> float:
        return float(self.params[_shift_name(f"ds{index}", self.config)].value)

    def background(self, index: int) -> float:
        return float(self.params[f"bkg_ds{index}"].value)

    def scatter(self, index: int) -> float:
        return float(self.params[f"scatter_ds{index}"].value)

    def curve(self, index: int) -> np.ndarray:
        return model_curve(self.params, index, self.datasets, self.irfs, self.config)

    def weighted_residuals(self, index: int) -> np.ndarray:
        return weighted_residuals(self.params, index, self.datasets, self.irfs, self.config)

    def irf_on_grid(self, index: int) -> np.ndarray:
        """The (shifted, normalised) IRF used for this dataset, on its time axis."""
        t_irf, irf = self.irfs[index]
        return shift_irf(self.datasets[index].time, irf, self.shift(index), t_irf)

    def reduced_chi2(self, index: int) -> float:
        """chi^2 / degrees of freedom for one dataset (shared parameters split evenly)."""
        r = self.weighted_residuals(index)
        free_per_dataset = self.fit.nvarys / len(self.datasets)
        return float(np.sum(r ** 2) / (r.size - free_per_dataset))

    def summary(self) -> pd.DataFrame:
        """One row per dataset: metadata, lifetimes, fractions, mean lifetimes, chi^2."""
        taus = self.taus
        rows = []
        for k, decay in enumerate(self.datasets):
            amps = self.amplitudes(k)
            row = {"name": decay.name, **decay.metadata}
            row.update({f"tau{i + 1}": tau for i, tau in enumerate(taus)})
            row.update({f"A{i + 1}": a for i, a in enumerate(amps)})
            row.update({f"a{i + 1}_frac": f for i, f in enumerate(fractions(amps, taus, "amplitude"))})
            row.update({f"f{i + 1}_frac": f for i, f in enumerate(fractions(amps, taus, "intensity"))})
            row.update(
                tau_amp=mean_lifetime(amps, taus, "amplitude"),
                tau_int=mean_lifetime(amps, taus, "intensity"),
                shift=self.shift(k),
                background=self.background(k),
                scatter=self.scatter(k),
                chi2_red=self.reduced_chi2(k),
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def report(self) -> str:
        """lmfit's text report, with the dataset label -> name mapping on top."""
        mapping = "\n".join(f"    {lbl} = {d.name}" for lbl, d in zip(self.labels, self.datasets))
        return f"[[Datasets]]\n{mapping}\n{fit_report(self.fit)}"


# --- entry point --------------------------------------------------------------------

def fit_global(datasets: Sequence[Decay], config: FitConfig) -> LifetimeResult:
    """Fit all ``datasets`` simultaneously with shared lifetimes."""
    datasets = list(datasets)
    if not datasets:
        raise ValueError("fit_global needs at least one dataset")

    irfs = prepare_irfs(datasets, config)

    shift_reference = None
    if config.shift_mode == "anchored":
        shift_reference = estimate_reference_shift(datasets, irfs, config)

    params = build_parameters(datasets, config, shift_reference=shift_reference)
    minimizer = Minimizer(objective, params, fcn_args=(datasets, irfs, config))
    fit = minimizer.minimize(method=config.method)
    return LifetimeResult(datasets, config, irfs, fit, shift_reference)
