"""
validate.py — Hold the pulse generator to published detector figures, and
plot what it produces.

A synthetic signal source is only admissible as a research instrument if it
can be shown to reproduce independently measured behaviour. This script is
that demonstration. Three checks:

  1. Pulse shape     — rise and decay time constants match the RC values set.
  2. Energy resolution — FWHM at 662 keV matches the ~6-7% reported for small
                          NaI(Tl)+PMT assemblies.
  3. Resolution scaling — FWHM(E) follows the 1/sqrt(E) statistical trend.

Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke

Run:  python3 validate.py
"""

import numpy as np
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pulsegen import (
    DetectorConfig,
    SourceConfig,
    PulseGenerator,
    compton_edge,
    codes_to_mv,
)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

INK = "#1a1a1a"
ACCENT = "#c1440e"
COOL = "#2e5c8a"


def fwhm_of_peak(values, centre_guess, window):
    """Gaussian-fit FWHM of a peak in a list of event amplitudes."""
    sel = values[np.abs(values - centre_guess) < window]
    mu, sigma = sel.mean(), sel.std()
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma, mu


# ==========================================================================
def check_1_pulse_shape(gen):
    det = gen.det
    t, v = gen.single_pulse(661.657, duration=400e-6)

    peak_i = int(np.argmax(v))
    peak_v = v[peak_i]

    # 10-90% rise time
    rise_region = v[: peak_i + 1]
    i10 = int(np.argmax(rise_region >= 0.10 * peak_v))
    i90 = int(np.argmax(rise_region >= 0.90 * peak_v))
    rise_ns = (i90 - i10) / det.sample_rate * 1e9

    # decay constant from the tail: fit ln(v) vs t well after the peak
    tail_start = peak_i + int(5 * det.tau_scint * det.sample_rate)
    tail_end = tail_start + int(2 * det.tau_preamp * det.sample_rate)
    tail_end = min(tail_end, len(v))
    tt = t[tail_start:tail_end]
    vv = v[tail_start:tail_end]
    good = vv > 0
    slope, _ = np.polyfit(tt[good], np.log(vv[good]), 1)
    tau_fit = -1.0 / slope

    print("--- Check 1: pulse shape ---")
    print(f"  peak amplitude          : {peak_v:8.2f} mV  (662 keV, noiseless)")
    # For a pure single-pole rise, 10-90% = tau_s * ln(9) = 2.197 * tau_s.
    # The measured value comes out slightly SHORTER because the tail is already
    # decaying while the pulse is still rising. That gap is physics, not a bug.
    print(f"  10-90% rise time        : {rise_ns:8.1f} ns")
    print(f"  single-pole ideal       : {np.log(9)*det.tau_scint*1e9:8.1f} ns  (upper bound)")
    print(f"  fitted decay constant   : {tau_fit*1e6:8.2f} us")
    print(f"  configured tau_preamp   : {det.tau_preamp*1e6:8.2f} us")

    # --- figure ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.3))

    m = t < 3e-6
    a1.plot(t[m] * 1e6, v[m], color=ACCENT, lw=1.6)
    a1.axhline(0, color=INK, lw=0.6)
    a1.set_xlabel("time (us)")
    a1.set_ylabel("preamp output (mV)")
    a1.set_title("Leading edge — set by scintillation decay", loc="left")
    a1.annotate(
        f"10-90% rise\n{rise_ns:.0f} ns",
        xy=(t[i90] * 1e6, v[i90]),
        xytext=(1.2, peak_v * 0.55),
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.8),
        fontsize=8,
    )

    a2.plot(t * 1e6, v, color=ACCENT, lw=1.2)
    a2.axhline(0, color=INK, lw=0.6)
    a2.set_xlabel("time (us)")
    a2.set_ylabel("preamp output (mV)")
    a2.set_title("Full tail pulse — decay set by Rf·Cf", loc="left")
    a2.annotate(
        f"fitted tau = {tau_fit*1e6:.1f} us\nset tau = {det.tau_preamp*1e6:.1f} us",
        xy=(0.55, 0.7),
        xycoords="axes fraction",
        fontsize=8,
    )

    fig.suptitle(
        "Figure 1 — Synthetic tail pulse, 662 keV, noiseless", y=0.98, fontsize=10
    )
    fig.tight_layout(rect=[0,0,1,0.93])
    fig.savefig("fig1_pulse_shape.png", bbox_inches="tight")
    plt.show()

# ==========================================================================
def check_2_trace(gen):
    t, codes, truth = gen.trace(duration=1.5e-3, rate_cps=20000)
    v = codes_to_mv(codes, gen.det)

    # Pile-up is only meaningful relative to the SHAPING time the DPP chain
    # will use (a few microseconds), not the preamp decay. Two events inside
    # one shaping window cannot be separated by amplitude alone.
    shaping_window = 10e-6

    print("\n--- Check 2: realistic digitised trace ---")
    print(f"  events in 1.5 ms        : {len(truth['arrival_times'])}")
    dt = np.diff(truth["arrival_times"])
    close = int((dt < shaping_window).sum())
    frac = 100.0 * close / max(len(dt), 1)
    print(f"  pairs within {shaping_window*1e6:.0f} us     : {close}  ({frac:.1f}%)")
    print("                            ^ pile-up, emerging from Poisson arrivals alone")
    print(f"  ADC codes span          : {codes.min()} .. {codes.max()} of {2**gen.det.adc_bits}")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.0, 4.6), sharex=False)

    a1.plot(t * 1e3, v, color=COOL, lw=0.6)
    for ti in truth["arrival_times"]:
        a1.axvline(ti * 1e3, color=ACCENT, lw=0.4, alpha=0.35)
    a1.set_ylabel("mV")
    a1.set_xlabel("time (ms)")
    a1.set_title(
        "Digitised preamp trace @ 20 kcps — red ticks are ground-truth arrivals",
        loc="left",
    )

    # zoom on the busiest millisecond-fraction to show pile-up
    if len(dt) and dt.min() < shaping_window:
        j = int(np.argmin(dt))
        t0 = truth["arrival_times"][j] - 20e-6
        t1 = truth["arrival_times"][j + 1] + 250e-6
    else:
        t0, t1 = 0.0, 300e-6
    m = (t >= t0) & (t <= t1)
    a2.plot(t[m] * 1e6, v[m], color=COOL, lw=1.0)
    for ti in truth["arrival_times"]:
        if t0 <= ti <= t1:
            a2.axvline(ti * 1e6, color=ACCENT, lw=0.8, alpha=0.6)
    a2.set_ylabel("mV")
    a2.set_xlabel("time (us)")
    a2.set_title("Zoom — pile-up: second pulse rides the first one's tail", loc="left")

    fig.suptitle("Figure 2 — What the ADC actually sees", y=0.98, fontsize=10)
    fig.tight_layout()
    fig.savefig("fig2_trace.png", bbox_inches="tight")
    plt.show()

# ==========================================================================
def check_3_resolution(gen):
    energies, amps = gen.event_amplitudes(400_000)

    peak_amp = np.median(amps[energies > 661.0])
    fw, mu = fwhm_of_peak(amps[energies > 661.0], peak_amp, peak_amp * 0.4)
    res_pct = 100.0 * fw / mu

    print("\n--- Check 3: intrinsic energy resolution ---")
    print(f"  FWHM at 662 keV         : {res_pct:8.2f} %")
    print(f"  published, small NaI(Tl): {'6 - 7':>8} %")
    verdict = "PASS" if 5.5 <= res_pct <= 7.5 else "OUT OF RANGE"
    print(f"  verdict                 : {verdict}")

    # spectrum
    kev_per_mv = 661.657 / mu
    spec_kev = amps * kev_per_mv
    hist, edges = np.histogram(spec_kev, bins=512, range=(0, 900))
    centres = 0.5 * (edges[1:] + edges[:-1])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    a1.step(centres, hist, where="mid", color=INK, lw=0.9)
    edge = compton_edge(661.657)
    a1.axvline(edge, color=COOL, ls="--", lw=1.0)
    a1.annotate(
        f"Compton edge\n{edge:.0f} keV",
        xy=(edge, hist.max() * 0.35),
        xytext=(edge - 300, hist.max() * 0.55),
        arrowprops=dict(arrowstyle="->", color=COOL, lw=0.8),
        fontsize=8,
        color=COOL,
    )
    a1.annotate(
        f"photopeak 662 keV\nFWHM = {res_pct:.1f}%",
        xy=(662, hist.max() * 0.9),
        xytext=(300, hist.max() * 0.85),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8),
        fontsize=8,
        color=ACCENT,
    )
    a1.set_xlabel("energy (keV)")
    a1.set_ylabel("counts / bin")
    a1.set_title("Simulated Cs-137 spectrum (detector only)", loc="left")

    # resolution vs energy
    test_E = np.array([60.0, 122.0, 356.0, 511.0, 661.657, 835.0, 1173.0, 1332.0])
    res = []
    for E in test_E:
        src = SourceConfig(gamma_energy=E, photopeak_fraction=1.0)
        g = PulseGenerator(det=gen.det, src=src, seed=7)
        _, a = g.event_amplitudes(60_000)
        f, m_ = fwhm_of_peak(a, np.median(a), np.median(a) * 0.5)
        res.append(100.0 * f / m_)
    res = np.array(res)

    a2.plot(test_E, res, "o-", color=ACCENT, lw=1.3, ms=4, label="model")
    ref = res[4] * np.sqrt(661.657 / test_E)
    a2.plot(test_E, ref, "--", color=COOL, lw=1.1, label=r"pure $1/\sqrt{E}$")
    a2.set_xlabel("energy (keV)")
    a2.set_ylabel("FWHM (%)")
    a2.set_title("Resolution vs energy", loc="left")
    a2.legend(frameon=False, fontsize=8)

    print("\n  FWHM vs energy:")
    for E, r in zip(test_E, res):
        print(f"    {E:8.1f} keV -> {r:5.2f} %")

    fig.suptitle(
        "Figure 3 — Detector resolution: the floor no DPP chain can beat",
        y=0.98,
        fontsize=10,
    )
    fig.tight_layout(rect=[0,0,1,0.93])
    fig.savefig("fig3_resolution.png", bbox_inches="tight")
    plt.show()

# ==========================================================================
if __name__ == "__main__":
    det = DetectorConfig()
    src = SourceConfig()
    gen = PulseGenerator(det=det, src=src, seed=42)

    check_1_pulse_shape(gen)
    check_2_trace(gen)
    check_3_resolution(gen)
    print("\nFigures written to fig{1,2,3}_*.png")
