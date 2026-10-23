"""Translate a FitConfig plus datasets into an lmfit ``Parameters`` set.

Naming (dataset ``k`` gets label ``dsk``):

    tau1..tauN          shared lifetimes
    A{i}_ds{k}          amplitude of component i in dataset k (always >= 0)
    bkg_ds{k}           constant background
    scatter_ds{k}       scatter (IRF-shaped) contribution
    shift               one IRF shift          (shift_mode = "global")
    shift_ds{k}         per-dataset IRF shift  (shift_mode = "per_dataset" / "anchored")
"""

from __future__ import annotations

from typing import Sequence

from lmfit import Parameters

from ..io import Decay
from .config import FitConfig, ParamSpec


def dataset_labels(n: int) -> list[str]:
    return [f"ds{k}" for k in range(n)]


def _add(params: Parameters, name: str, spec: ParamSpec) -> None:
    params.add(name, value=spec.value, min=spec.min, max=spec.max, vary=spec.vary)


def amplitude_guesses(decay: Decay, config: FitConfig) -> list[float]:
    """Start amplitudes. With a unit-sum IRF, total counts ~= sum(A_i * tau_i)."""
    total = float(decay.counts.sum())
    return [factor * total / (config.n_exp * tau.value)
            for factor, tau in zip(config.amplitude_guess_factors, config.taus)]


def build_parameters(datasets: Sequence[Decay], config: FitConfig,
                     shift_reference: float | None = None) -> Parameters:
    params = Parameters()

    for i, spec in enumerate(config.taus, start=1):
        _add(params, f"tau{i}", spec)

    if config.shift_mode == "global":
        _add(params, "shift", config.shift)
    elif config.shift_mode == "anchored" and shift_reference is None:
        raise ValueError("shift_mode='anchored' needs a shift_reference (fit_global computes it)")

    for label, decay in zip(dataset_labels(len(datasets)), datasets):
        for i, guess in enumerate(amplitude_guesses(decay, config), start=1):
            params.add(f"A{i}_{label}", value=guess, min=0.0)

        _add(params, f"bkg_{label}", config.background)
        _add(params, f"scatter_{label}", config.scatter)

        if config.shift_mode == "per_dataset":
            _add(params, f"shift_{label}", config.shift)
        elif config.shift_mode == "anchored":
            w = config.anchor_window
            params.add(f"shift_{label}", value=shift_reference,
                       min=shift_reference - w, max=shift_reference + w)

    return params
