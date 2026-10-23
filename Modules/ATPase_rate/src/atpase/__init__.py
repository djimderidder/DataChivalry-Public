"""atpase: ATPase rates from NADH-coupled FLUOstar absorbance assays (340 nm).

Modules
-------
settings   AtpaseSettings: extinction coefficient, path length, enzyme column, auto-window
signal     well minus control well
windows    fitting window: automatic (NADH depletion), from the config (Starti/Endi), or both
rate       linear fit -> ATPase rate (ATP per enzyme per minute)
batch      analyse_config (all config rows), update_config (write picked windows back)
plotting   figures (returned, never shown here)
synthetic  mock plates (+ config) written as FLUOstar CSV exports
"""

from .batch import analyse_config, update_config
from .rate import AtpaseFit, atpase_rate
from .settings import AtpaseSettings
from .signal import background_subtracted
from .windows import auto_stop, fit_window

__all__ = ["AtpaseSettings", "AtpaseFit", "atpase_rate", "background_subtracted", "auto_stop",
           "fit_window", "analyse_config", "update_config"]
