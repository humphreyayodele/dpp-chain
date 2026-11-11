# gamma-dpp-chain

A physics-based simulation framework and a software digital pulse processing (DPP) chain for low-cost NaI(Tl) gamma-ray spectrometry. This is the code and data behind the paper:

> U. Anekwe, H. A. Peleyeju and C. Emeruwa, *A physics-based simulation framework for designing and testing digital pulse processing chains for low-cost NaI(Tl) gamma-ray spectrometry: description and verification*, [journal, year], [DOI] — to be completed on acceptance

Archived version: DOI: 10.5281/zenodo.24147429

## What is here

- `pulsegen.py` is the signal model. It generates digitised preamplifier traces from a NaI(Tl) scintillator and photomultiplier with modelled noise, mains pickup and ADC quantisation. It also returns the true arrival time and energy of every event (the ground truth).
- `dpp.py` is the processing chain:
  - pole-zero corrected trapezoidal shaping (Jordanov–Knoll recursion);
  - a gated moving-average baseline restorer;
  - a fast trigger with hysteresis re-arm and second-rise pile-up detection;
  - pile-up rejection and amplitude measurement.
- The other scripts produce the numbers and figures in the paper (table below). Each one runs in the folder beside `pulsegen.py` and `dpp.py` and writes its output there.

**Scope.** The paper describes the framework and verifies it against its design: pulse shape, calibration, amplitude calibration of the shaper, and low-frequency behaviour of the shaper and the baseline restorer. All results are obtained from modelled signals; the chain has not been tested on waveforms recorded from a physical detector. The 6.5 % resolution at 662 keV is the model's calibration target and is not an independent test.

The measured limits and performance of the chain (trigger and pile-up detection on close pairs, pile-up rejection against count rate, the restorer with drift and with pulses, resolution, cost) are reported separately. Their scripts and data are not in this repository.

**Note on the comments in `dpp.py`.** The file is the chain exactly as used in the wider study, so its docstrings summarise some of those separately reported results and name scripts that are not part of this release (`trig_check.py`, `fig6_v2.py`, `linearity.py`, `blr_check.py`, `d1_diagnose.py`, `d1_retest.py`, `d1_blindspot.py`, `pileup_model_check.py`, and Part C2 of `ramp_test.py`). Nothing in this repository depends on them.

## Requirements

Python 3.13 and the packages pinned in `requirements.txt` (the versions the published results were produced with):

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   or   source .venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt
```

All random streams are seeded. With the pinned versions, every script reproduces the published values exactly.

## Which script gives which result

| Paper result | Script | Output |
|---|---|---|
| Fig. 1 (block diagram of the framework) | `paper1_figs.py` | PNG, PDF |
| Figs. 2–4 (chain stages, pulse shape, detector resolution) | `thesis_figs.py` (uses `validate.py`, `run_chain.py`) | PNG |
| Fig. 5 (mis-set pole-zero constant) | `run_chain.py` | PNG, PDF |
| Section 2.1.3, headroom and clipping | `c3_headroom.py`, `c2_stacked_level.py` | `c2_stacked_level.csv`, `c2_event_level.csv` |
| Section 3.3, amplitude calibration and exact M | `gain_check.py`, `appendix_a_check.py` | printed |
| Section 3.5, low-frequency behaviour of the shaper and restorer | `ramp_test.py` (Part A), `pullia_validation.py`, `trailing_window_check.py` | `pullia_validation.csv`, `trailing_window_check.csv` |
| Sections 3.5–3.6, restorer gate and shaped mains residue | `ramp_test.py` (Part C) | printed |

`c2_stacked_level.py` is the longest run. The other scripts finish in seconds to a few minutes.

## Licensing

- Source code: the Python scripts are licensed under the MIT License (`LICENSE`).
- Data: the CSV result files are licensed under the Creative Commons Attribution 4.0 International License (`LICENSE-DATA`).
- The figures are not stored here; the scripts regenerate them. The paper's text and figures are covered by the journal's publishing terms.

## Citing

Please cite the paper and the archived version of this repository (`CITATION.cff` has the details; GitHub's "Cite this repository" button reads it).

## Use of AI assistance

Some of the analysis and plotting scripts were written with the help of an AI assistant (Claude, Anthropic). The signal model and processing chain were designed and verified by H. A. Peleyeju, and every script was run and checked on his computer.

## Author of the code

Humphrey Ayodele Peleyeju — ORCID 0009-0004-9043-9072
