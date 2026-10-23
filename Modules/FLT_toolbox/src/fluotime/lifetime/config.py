"""Fit configuration: which model, which constraints, which options.

A config can be built in Python, from a dict, or from a TOML/JSON file (see
``configs/`` for examples). Every option that differed between the original
scripts is a field here, so all of those variants are one code path.
"""

from __future__ import annotations

import dataclasses
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IRF_SOURCES = ("reference", "own")
SHIFT_MODES = ("global", "per_dataset", "anchored")
IRF_PREPROCESSING = ("mean_threshold", "percentile_subtract", "none")
WEIGHTINGS = ("neyman", "pearson", "none")
MAX_EXPONENTIALS = 4


@dataclass
class ParamSpec:
    """Starting value and constraints of one fit parameter."""

    value: float
    min: float = -math.inf
    max: float = math.inf
    vary: bool = True

    def __post_init__(self):
        self.value, self.min, self.max = float(self.value), float(self.min), float(self.max)
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) is larger than max ({self.max})")
        if self.vary and not (self.min <= self.value <= self.max):
            raise ValueError(f"start value {self.value} is outside [{self.min}, {self.max}]")

    @classmethod
    def from_any(cls, spec: "ParamSpec | float | dict") -> "ParamSpec":
        if isinstance(spec, ParamSpec):
            return spec
        if isinstance(spec, dict):
            return cls(**spec)
        return cls(value=spec)


@dataclass
class FitConfig:
    """All settings of a global lifetime fit.

    Model
        n_exp                    number of exponentials (1-4); lifetimes are shared by all datasets
        taus                     one ParamSpec (or number / dict) per lifetime
        amplitude_guess_factors  start amplitude_i = factor_i * total_counts / (n_exp * tau_i)
        shift, background, scatter
                                 ParamSpecs; background and scatter are applied per dataset
    Options
        irf_source       "reference": the IRF of datasets[reference_index] for all datasets
                         "own": every dataset uses its own IRF
        shift_mode       "global": one IRF shift for all datasets
                         "per_dataset": a free shift per dataset (bounds from ``shift``)
                         "anchored": fit the reference dataset alone first, then let each
                                     dataset's shift vary within +/- anchor_window of it
        irf_preprocessing  see fluotime.irf ("mean_threshold" is what the original code did)
        weighting        "neyman" 1/sqrt(max(data,1)) (original), "pearson" 1/sqrt(max(model,1)),
                         or "none"
        method           lmfit minimiser name (default "leastsq")
    """

    n_exp: int
    taus: list[Any]
    amplitude_guess_factors: list[float] | None = None
    shift: Any = field(default_factory=lambda: ParamSpec(0.0, -1.0, 1.0))
    background: Any = field(default_factory=lambda: ParamSpec(2.5, 0.0, 100.0))
    scatter: Any = field(default_factory=lambda: ParamSpec(3000.0, 0.0, 1e5))
    irf_source: str = "reference"
    reference_index: int = 0
    irf_preprocessing: str = "mean_threshold"
    irf_percentile: float = 5.0
    shift_mode: str = "global"
    anchor_window: float = 0.05
    weighting: str = "neyman"
    method: str = "leastsq"

    def __post_init__(self):
        if not 1 <= int(self.n_exp) <= MAX_EXPONENTIALS:
            raise ValueError(f"n_exp must be between 1 and {MAX_EXPONENTIALS}, got {self.n_exp}")
        self.n_exp = int(self.n_exp)
        if len(self.taus) != self.n_exp:
            raise ValueError(f"got {len(self.taus)} tau specs for n_exp={self.n_exp}")
        self.taus = [ParamSpec.from_any(t) for t in self.taus]

        if self.amplitude_guess_factors is None:
            self.amplitude_guess_factors = [1.0] * self.n_exp
        if len(self.amplitude_guess_factors) != self.n_exp:
            raise ValueError("amplitude_guess_factors needs one value per exponential (n_exp)")
        self.amplitude_guess_factors = list(self.amplitude_guess_factors)

        self.shift = ParamSpec.from_any(self.shift)
        self.background = ParamSpec.from_any(self.background)
        self.scatter = ParamSpec.from_any(self.scatter)

        for name, allowed in (("irf_source", IRF_SOURCES), ("shift_mode", SHIFT_MODES),
                              ("irf_preprocessing", IRF_PREPROCESSING), ("weighting", WEIGHTINGS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r} is not one of {allowed}")
        if self.anchor_window <= 0:
            raise ValueError("anchor_window must be positive")

    # --- construction helpers ---------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "FitConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}; allowed: {sorted(known)}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "FitConfig":
        """Load a ``.toml`` or ``.json`` file."""
        path = Path(path)
        if path.suffix == ".toml":
            data = tomllib.loads(path.read_text())
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
        else:
            raise ValueError(f"config file must be .toml or .json, got {path.suffix}")
        return cls.from_dict(data)

    def with_overrides(self, **changes) -> "FitConfig":
        """Copy with some fields changed (handy in runFluoTime.py to compare options)."""
        return dataclasses.replace(self, **changes)
