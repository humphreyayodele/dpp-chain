"""
pullia_validation.py - low-frequency response of the shaper and of the gated
baseline restorer, at the trapezoid geometry of Pullia, Gritti & Ripamonti,
IEEE Trans. Nucl. Sci. 44(3) (1997) 331-337, doi:10.1109/23.603665.

Their Section V used a trapezoidal weight function of 10 us total width with a
1.6 us flat top, and swept an injected sinusoid from 50 Hz to 4 kHz, comparing
the gain with and without the digital restorer (their Figs. 10-12). This script
reproduces that measurement on this study's chain.

Two measurements:
  1. Shaper alone. Output amplitude per 1 mV input sinusoid, confirming the
     plateau at l/(M+1) and the rise as |1 + j2*pi*f*tau|.
  2. Shaper + gated moving-average restorer (200 us window), as the ratio of
     output amplitude with the restorer to that without, in dB.

The restorer gate is held open (threshold = inf) so that measurement 2 is the
linear transfer function, which is the quantity Pullia plot. That is justified
by a separate run with modelled noise and the chain's own 3-sigma gate, in
which the gate is open for 99.8% of samples.

Comparison curve: Pullia's own low-frequency model, in which each restorer lobe
is replaced by a delta of equal area at its barycentre (they state the
low-frequency behaviour is independent of lobe shape). For two lobes placed
symmetrically at +/- 200 us, that gives 1 - cos(2*pi*f*200us). Their figures have NOT been
digitised here; the comparison is against their analytic model.

Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke

Run:  python pullia_validation.py      (about 30 s)
"""
import numpy as np, pandas as pd
from dpp import trapezoidal_filter, pz_parameter, filter_gain, restore_baseline, mad_sigma

FS, TAU = 50e6, 50e-6
M = pz_parameter(TAU, FS)
FT = 1.6e-6                       # Pullia Sec. V flat top
TP = (10e-6 - FT) / 2             # 10 us total width = 2*TP + FT
K = int(round(TP * FS)); L = K + int(round(FT * FS)); G = filter_gain(K, M)
WIN = int(200e-6 * FS)            # this chain's restorer window
W = WIN / FS
NC, MARGIN = 4, 3e-3              # cycles measured, and edge trimmed each side


def lockin(x, f, t, sl):
    return 2.0 * abs((x[sl] * np.exp(-2j * np.pi * f * t[sl])).mean())


def sweep(freqs):
    rows = []
    for f in freqs:
        n = int(round((NC / f + 2 * MARGIN) * FS))
        t = np.arange(n) / FS
        v = np.sin(2 * np.pi * f * t)
        s_raw = trapezoidal_filter(v, K, L, M)
        s_blr, _ = restore_baseline(s_raw, WIN, np.inf)
        e = int(MARGIN * FS)
        sl = slice(e, e + int(round(NC * FS / f)))
        a_raw = lockin(s_raw, f, t, sl) / G
        a_blr = lockin(s_blr, f, t, sl) / G
        rows.append(dict(freq_Hz=f, shaper_mV_per_mV=a_raw, with_blr_mV_per_mV=a_blr,
                         gain_dB_measured=20 * np.log10(a_blr / a_raw),
                         gain_dB_pred_1_minus_sinc=20 * np.log10(abs(1 - np.sinc(f * W))),
                         gain_dB_pullia_2lobe_model=20 * np.log10(abs(1 - np.cos(2 * np.pi * f * 200e-6)))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"k = {K}, l = {L}, M = {M:.4f}, l/(M+1) = {L/(M+1):.5f}, window = {W*1e6:.0f} us")
    df = sweep(np.unique(np.round(np.logspace(np.log10(50), np.log10(10000), 18))))
    df.to_csv("pullia_validation.csv", index=False)
    resid = (df.gain_dB_measured - df.gain_dB_pred_1_minus_sinc).abs().max()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nmax |measured - 1-sinc(fW)| = {resid:.4f} dB")
    print(f"at 50 Hz: this chain {df.gain_dB_measured.iloc[0]:.1f} dB, "
          f"Pullia two-lobe model {df.gain_dB_pullia_2lobe_model.iloc[0]:.1f} dB")
