"""thtdecay: ThT disaggregation kinetics from FLUOstar plate-reader data.

Modules
-------
settings   ThtSettings: gain, trimming, windows, smoothing (replaces the old ``Mode``)
normalize  normalize_tht: control-based normalisation of ThT traces
kinetics   prepare, tht_kinetics, process_well, analyse_config (batch from a spreadsheet)
plotting   figures (returned, never shown here)
synthetic  power-law mock plates (+ config) written as FLUOstar CSV exports
"""

from .kinetics import ThtKinetics, analyse_config, prepare, process_well, tht_kinetics
from .normalize import normalize_tht
from .settings import ThtSettings

__all__ = ["ThtSettings", "ThtKinetics", "normalize_tht", "prepare", "tht_kinetics",
           "process_well", "analyse_config"]
