# DataChivalry

DataChivalry is a collection of data analysis tools developed during my PhD at the Wentink Lab (Leiden University), where I study the role of HSP70 chaperones in protein disaggregation.
As a hobby project, I aim to make many of the analysis scripts I have written for my own research, as well as for collaborators, available to the broader scientific community. I am currently transferring and restructuring different modules, with the goal of later making them more general, user-friendly, and widely applicable.

## Modules

Each module contains its own dedicated README with additional documentation and usage examples.

* `ATPase\_rate`  
Calculate ATPase activity rates from FLUOstar absorbance measurements.
* `Chromatogram\_fitter`  
Fit user-defined distributions to chromatogram traces from ÄKTA and NGC systems. *(Intentionally limited public version: related work is covered by a patent application.)*
* `EPR\_fitting`  
Simulate EPR spectra from spin systems. *(Code currently only available upon request.)*
* `FLT\_toolbox`  
Global fluorescence lifetime fitting and steady-state anisotropy for FluoTime data. *(Archived.)*
* `Gel\_quantification`  
Quantification of band intensities from Western blots and gel electrophoresis images.
* `Imaging`  
Image analysis and segmentation of fibril structures from electron microscopy (EM) images.
* `ThT\_decay`  
Processing and analysis of high-throughput Thioflavin T fluorescence measurements from FLUOstar plate readers.

`ATPase\_rate` and `ThT\_decay` share the FLUOstar helpers in `Shared/` (loading exports, time windows, config spreadsheets, well selection); their run scripts find them automatically.

## Running and testing

Most modules have run scripts with a **USER SETTINGS** block at the top: fill it in and run the file
(e.g. Spyder: *Run file*). The same scripts also run from a terminal, where flags override the settings.
`ATPase\_rate`, `ThT\_decay`, `FLT\_toolbox` and `Gel\_quantification` run on simulated data when no data is given, so they can be tried out first.
Modules with a `tests/` folder are tested with `pytest` (run it inside the module folder); tests on real
data are skipped unless you put your own files in that module's `tests/data/real/`, which is git-ignored.

## Status

This repository is a work in progress. Modules are being actively migrated, cleaned up, documented, and generalized for broader use.
Contributions, suggestions, and feedback are always welcome.

