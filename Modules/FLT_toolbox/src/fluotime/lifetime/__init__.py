"""Global multi-exponential lifetime fitting.

    from fluotime.lifetime import FitConfig, fit_global
    cfg = FitConfig.from_file("configs/biexp_af488.toml")
    result = fit_global(decays, cfg)
    print(result.report())
    result.summary()
"""

from .config import FitConfig, ParamSpec
from .fit import LifetimeResult, fit_global
from .results import fractions, mean_lifetime

__all__ = ["FitConfig", "ParamSpec", "LifetimeResult", "fit_global", "fractions", "mean_lifetime"]
