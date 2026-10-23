"""fluotime: example analysis code for FluoTime fluorescence data.

Modules
-------
io          load decay (.dat: time, decay, IRF) and anisotropy spectra
irf         IRF background handling, normalisation and time shifting
models      pure reconvolution model (sum of exponentials convolved with the IRF)
lifetime    global multi-exponential fitting (config, parameters, fit, results)
anisotropy  steady-state anisotropy spectra (G factor, blank subtraction, smoothing)
plotting    matplotlib figures for both analyses
synthetic   simulated data, used by the tests and the demo in runFluoTime.py
"""

from .io import Decay, Spectrum, load_decay, load_spectrum

__all__ = ["Decay", "Spectrum", "load_decay", "load_spectrum"]
