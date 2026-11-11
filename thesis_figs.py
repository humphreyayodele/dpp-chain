"""
thesis_figs.py - regenerate Figures 3.1, 3.2 and 3.3 for Chapter 3.

Calls Ayodele's own plotting functions unchanged (validate.py, run_chain.py)
with two differences, both made from outside, so none of his files are edited:
  1. the baked-in old titles ("Figure 1 -", "Figure 3 -", "Figure 4 -") are
     suppressed, because the thesis captions carry the numbering;
  2. Figure 3.3 is drawn twice: once with the dpp.py defaults (as before) and
     once with the D1 settings that Chapter 3 now describes,
     ChainConfig(fast_rearm_sigma=4.0, fast_pileup_sigma=6.0).

Put this file beside pulsegen.py, dpp.py, validate.py and run_chain.py, then:
    python thesis_figs.py
Outputs go to a new subfolder, thesis_figs/, so the old PNGs are not touched.
Saved at 300 dpi for print.

Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke
"""
import os
import matplotlib
matplotlib.use("Agg")                      # write files, no windows to close
from matplotlib.figure import Figure
Figure.suptitle = lambda self, *a, **k: None   # drop the old "Figure N" titles

import validate
import run_chain
from pulsegen import DetectorConfig, SourceConfig, PulseGenerator
from dpp import ChainConfig

# print resolution: the scripts set figure.dpi (screen) but not savefig.dpi
matplotlib.rcParams["savefig.dpi"] = 300

os.makedirs("thesis_figs", exist_ok=True)
os.chdir("thesis_figs")

# Figures 3.1 and 3.2 - same set-up as validate.py's __main__
det = DetectorConfig()
gen = PulseGenerator(det=det, src=SourceConfig(), seed=42)
validate.check_1_pulse_shape(gen)            # -> fig1_pulse_shape.png
os.replace("fig1_pulse_shape.png", "Figure_3_1_pulse_shape.png")
validate.check_3_resolution(gen)             # -> fig3_resolution.png
os.replace("fig3_resolution.png", "Figure_3_2_resolution.png")

# Figure 3.3 - same set-up as run_chain.py's __main__, drawn twice
print("\n=== Figure 3.3, dpp.py defaults (earlier chain) ===")
run_chain.fig4_stages(det, PulseGenerator(det=det, seed=42))
os.replace("fig4_stages.png", "Figure_3_3_stages_default.png")

print("\n=== Figure 3.3, D1 settings (chain described in Chapter 3) ===")
run_chain.ChainConfig = lambda **kw: ChainConfig(
    fast_rearm_sigma=4.0, fast_pileup_sigma=6.0, **kw)
run_chain.fig4_stages(det, PulseGenerator(det=det, seed=42))
os.replace("fig4_stages.png", "Figure_3_3_stages_D1.png")

print("\nWritten to thesis_figs/: Figure_3_1_pulse_shape.png, "
      "Figure_3_2_resolution.png, Figure_3_3_stages_default.png, "
      "Figure_3_3_stages_D1.png")
