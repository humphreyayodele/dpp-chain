"""
trailing_window_check.py - what a real-time (causal) baseline restorer would lose.

Item B13 of Ayodele/ch1_3_impact_verification.md. Chapter 3 (Sec. 3.3) records that
this chain's restorer uses a CENTRED 200 us window (100 us before and 100 us after
each sample), so it needs future samples and only works offline; real-time
hardware would need a TRAILING window (the 200 us before each sample only).
This script measures what that change costs in low-frequency rejection.

Same method as pullia_validation.py (same 10 us / 1.6 us trapezoid, same lock-in
sweep, gate held open so the result is the restorer's linear transfer function).
Two restorers, same 200 us length:
  centred  - dpp.restore_baseline (the chain as it is)
  trailing - mean of the 200 us of shaped output BEFORE each sample (written here;
             dpp.py is not changed)

Prints the gain (dB) of each restorer against frequency. No model columns are
printed on purpose: write your prediction down first (gain at 50 Hz, and the
slope in dB per decade, for each restorer), then compare.

Run:  python trailing_window_check.py      (about 30 s; needs dpp.py beside it)
Writes trailing_window_check.csv.

Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke
"""
import numpy as np, pandas as pd
from dpp import trapezoidal_filter, pz_parameter, filter_gain, restore_baseline

FS, TAU = 50e6, 50e-6
M = pz_parameter(TAU, FS)
FT = 1.6e-6
TP = (10e-6 - FT) / 2
K = int(round(TP * FS)); L = K + int(round(FT * FS)); G = filter_gain(K, M)
WIN = int(200e-6 * FS)
NC, MARGIN = 4, 3e-3


def restore_trailing(s, win):
    """Subtract the mean of the `win` samples before each sample (gate open)."""
    c = np.concatenate(([0.0], np.cumsum(s)))
    n = np.arange(len(s))
    lo = np.clip(n - win, 0, None)
    base = (c[n] - c[lo]) / np.maximum(n - lo, 1)
    return s - base


def lockin(x, f, t, sl):
    return 2.0 * abs((x[sl] * np.exp(-2j * np.pi * f * t[sl])).mean())


def sweep(freqs):
    rows = []
    for f in freqs:
        n = int(round((NC / f + 2 * MARGIN) * FS))
        t = np.arange(n) / FS
        s_raw = trapezoidal_filter(np.sin(2 * np.pi * f * t), K, L, M)
        s_cen, _ = restore_baseline(s_raw, WIN, np.inf)
        s_tr = restore_trailing(s_raw, WIN)
        e = int(MARGIN * FS)
        sl = slice(e, e + int(round(NC * FS / f)))
        a_raw = lockin(s_raw, f, t, sl)
        rows.append(dict(freq_Hz=f,
                         centred_dB=20 * np.log10(lockin(s_cen, f, t, sl) / a_raw),
                         trailing_dB=20 * np.log10(lockin(s_tr, f, t, sl) / a_raw)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"k = {K}, l = {L}, window = {WIN / FS * 1e6:.0f} us (both restorers)")
    df = sweep(np.array([50, 100, 200, 500, 1000, 2000, 5000]))
    df.to_csv("trailing_window_check.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    for col in ("centred_dB", "trailing_dB"):
        slope = df[col].iloc[1] - df[col].iloc[0]          # 50 -> 100 Hz is one octave
        print(f"{col}: change from 50 to 100 Hz = {slope:+.1f} dB per octave "
              f"(x 3.32 = {slope * 3.32:+.0f} dB per decade)")
