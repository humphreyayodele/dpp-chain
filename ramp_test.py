"""
ramp_test.py - how this study's shaping recursion treats a constant offset,
a straight-line drift and mains pickup; and how large the shaped mains residue
is against the baseline restorer's gate.
 
Put beside dpp.py and pulsegen.py, then:   python ramp_test.py
Uses dpp.py's own trapezoidal_filter and mad_sigma (unedited).
Takes about a minute.
 
Release version (11 Nov 2026): Part A of the thesis script and the gate and
mains-residue columns of its Part C, which are the parts the paper uses. The thesis script's comparison
with another group's published recursion is not part of this release.
Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke 30 Oct 2026
"""
import numpy as np
from dpp import trapezoidal_filter, pz_parameter, filter_gain, mad_sigma
from pulsegen import DetectorConfig, PulseGenerator, codes_to_mv
 
FS, TAU, FT = 50e6, 50e-6, 1e-6          # this study: 50 MS/s, 50 us, 1 us flat top
M = pz_parameter(TAU, FS)
 
 
def ours(v, tp):
    """This study's filter, output in input-mV units (divided by k(M+1))."""
    k = int(round(tp * FS)); l = k + int(round(FT * FS))
    return trapezoidal_filter(v, k, l, M) / filter_gain(k, M), k, l
 
 
def at(ms, fs):
    return int(ms * 1e-3 * fs) - 1
 
 
print("Part A - this study's filter (4 us peaking, 1 us flat top), no noise")
n = np.arange(int(4e-3 * FS)).astype(float)
s, k, l = ours(np.ones_like(n), 4e-6)
print(f"  A1 constant 1 mV from t = 0: output at 0.1, 1, 4 ms = "
      f"{s[at(0.1, FS)]:.5f}, {s[at(1, FS)]:.5f}, {s[at(4, FS)]:.5f} mV   "
      f"[l/(M+1) = {l/(M+1):.5f}]")
s16, k16, l16 = ours(np.ones_like(n), 16e-6)
print(f"     same at 16 us peaking: {s16[at(4, FS)]:.5f} mV   [l/(M+1) = {l16/(M+1):.5f}]")
drift = 1.0 / (1e-3 * FS)                          # 1 mV per ms, in mV per sample
s, _, _ = ours(drift * n, 4e-6)
for t in (1, 2, 4):
    print(f"  A2 drift 1 mV/ms: at {t} ms input {drift * (at(t, FS) + 1):.3f} mV, "
          f"output {s[at(t, FS)]:.4f} mV")
t = np.arange(int(60e-3 * FS)) / FS
for f in (50, 200, 1000):
    s, _, _ = ours(4.0 * np.sin(2 * np.pi * f * t), 4e-6)
    tail = s[len(s) // 2:]
    pred = 4.0 * l / (M + 1) * np.sqrt(1 + (2 * np.pi * f * TAU) ** 2)
    print(f"  A3 4 mV sine at {f:4d} Hz: output amplitude {(tail.max() - tail.min()) / 2:.4f} mV"
          f"   [predicted {pred:.4f}]")
 

print("\nPart C - the restorer's gate and the shaped mains residue on modelled noise "
      "(no pulses, 40 ms, seed 3)")
print("  front end   peak  gate(3 sigma)  mains residue  quiet %   (mV-equiv)")
fronts = [("nominal", DetectorConfig()),
          ("low-cost", DetectorConfig(noise_rms_mv=55.0, parallel_noise_rms_mv=30.0))]
for name, det in fronts:
    for tp in (4e-6, 16e-6):
        _, c, _ = PulseGenerator(det=det, seed=3).trace(40e-3, rate_cps=0)
        v = codes_to_mv(c, det)
        kk = int(round(tp * FS)); ll = kk + int(round(FT * FS)); g = filter_gain(kk, M)
        s_raw = trapezoidal_filter(v, kk, ll, M)
        thr = 3.0 * mad_sigma(s_raw)            # as DPPChain.process does
        quiet = 100 * (np.abs(s_raw) < thr).mean()
        gate = thr / g
        resid = det.mains_amplitude_mv * ll / (M + 1) * np.sqrt(1 + (2 * np.pi * 50 * TAU) ** 2)
        print(f"  {name:9s} {tp*1e6:4.0f} us  {gate:9.3f}      {resid:9.3f}     {quiet:5.1f}")
