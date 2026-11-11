"""
run_chain.py — Drive the DPP chain and show what each stage does.

  Fig 4 — the chain stage by stage on one trace. Shows what each operator does.
  Fig 5 — spectrum out of the chain, against the detector floor from Fig 3.
  Fig 7 — pole-zero mismatch: what a wrong τ does to the trapezoid.

Fig 6 (FWHM vs peaking time, the central result) now lives in fig6_v2.py.
The energy-linearity test lives in linearity.py.

Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke

Run:  python run_chain.py
"""

import time
import numpy as np
import matplotlib

# matplotlib.use("Agg")   # uncomment for headless / file-only output
import matplotlib.pyplot as plt

from pulsegen import (DetectorConfig, SourceConfig, PulseGenerator, codes_to_mv,
                      adc_full_scale_mv)
from dpp import ChainConfig, DPPChain, trapezoidal_filter

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
INK, ACCENT, COOL, GREEN = "#1a1a1a", "#c1440e", "#2e5c8a", "#2d6a4f"
FW = 2.0 * np.sqrt(2.0 * np.log(2.0))


def fwhm_pct(amps, clip_sigma=5.0):
    """FWHM of a peak, as a percent of its centroid.

    One MAD-based pass to drop outliers, then one wide (5-sigma) mean/std
    pass. No iterative narrowing: the earlier version re-clipped at 2.5 sigma
    three times and understated FWHM (6.10% reported for a true 6.50%).
    Same estimator as fig6_fixed.py and fig6_v2.py.
    """
    amps = np.asarray(amps)
    if len(amps) < 50:
        return np.nan, np.nan
    med = np.median(amps)
    mad = np.median(np.abs(amps - med)) * 1.4826
    sel = amps[np.abs(amps - med) < clip_sigma * mad] if mad > 0 else amps
    if len(sel) < 50:
        return np.nan, np.nan
    mu, sd = sel.mean(), sel.std()
    sel = amps[np.abs(amps - mu) < clip_sigma * sd]
    mu, sd = sel.mean(), sel.std()
    return 100.0 * FW * sd / mu, mu


# ==========================================================================
def fig4_stages(det, gen):
    cfg = ChainConfig(peaking_time=4e-6, flat_top=1.0e-6, tau_preamp=det.tau_preamp)
    ch = DPPChain(cfg, det.sample_rate)

    t, codes, truth = gen.trace(duration=600e-6, rate_cps=8000)
    v = codes_to_mv(codes, det)
    out = ch.process(v, v_max=adc_full_scale_mv(det))

    print("--- Fig 4: chain stages ---")
    print(f"  peaking time            : {cfg.peaking_time*1e6:.1f} us")
    print(f"  k, l, M                 : {ch.k}, {ch.l}, {ch.M:.1f}")
    print(f"  fast filter k_f, l_f    : {ch.kf}, {ch.lf}")
    print(f"  true events in trace    : {len(truth['arrival_times'])}")
    print(f"  triggered               : {out['n_triggered']}")
    print(f"  rejected as pile-up     : {out['n_rejected']}")
    print(f"  rejected, ADC clipped   : {out['n_saturated']}")
    print(f"  measured                : {len(out['amplitudes'])}")

    fig, axs = plt.subplots(3, 1, figsize=(9.5, 6.2), sharex=True)

    axs[0].plot(t * 1e6, v, color=COOL, lw=0.7)
    axs[0].set_ylabel("mV")
    axs[0].set_title(
        "1 — Raw digitised trace: exponential tails stack, baseline climbs", loc="left"
    )

    axs[1].plot(t * 1e6, out["shaped_raw"] / ch.gain, color=ACCENT, lw=0.8)
    axs[1].plot(t * 1e6, out["baseline"] / ch.gain, color=INK, lw=1.0, ls="--",
                label="estimated baseline")
    axs[1].set_ylabel("mV (equiv.)")
    axs[1].set_title(
        "2 — After pole-zero + trapezoidal shaping: tails gone, flat tops appear",
        loc="left",
    )
    axs[1].legend(frameon=False, fontsize=8, loc="upper right")

    axs[2].plot(t * 1e6, out["shaped"] / ch.gain, color=GREEN, lw=0.8,
                label="slow (energy) filter")
    axs[2].plot(t * 1e6, out["fast"], color=COOL, lw=0.6, alpha=0.8,
                label="fast (timing) filter")
    thr = cfg.fast_trigger_sigma * out["fast_sigma"]
    axs[2].axhline(thr, color=INK, ls=":", lw=0.9,
                   label=f"fast trigger ({cfg.fast_trigger_sigma:.0f}σ)")
    for i in out["starts"]:
        axs[2].axvline(t[i] * 1e6, color=ACCENT, lw=0.5, alpha=0.5)
    axs[2].set_ylabel("mV (equiv.)")
    axs[2].set_xlabel("time (us)")
    axs[2].set_title(
        "3 — Baseline restored; fast filter times each pulse (lines); pile-up removed",
        loc="left",
    )
    axs[2].legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Figure 4 — The chain, stage by stage", y=0.98, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fig4_stages.png", bbox_inches="tight")
    plt.show()


# ==========================================================================
def collect_spectrum(det, src, cfg, seed=11, n_traces=90, trace_len=12e-3, rate=8000):
    """Run many traces through the chain, return measured amplitudes.

    "seconds" times the CHAIN ONLY (ch.process), not pulse generation.
    """
    gen = PulseGenerator(det=det, src=src, seed=seed)
    ch = DPPChain(cfg, det.sample_rate)
    vmax = adc_full_scale_mv(det)
    amps, n_true, n_rej, n_trig, n_sat = [], 0, 0, 0, 0
    dt = 0.0
    n_samples = 0
    for _ in range(n_traces):
        _, codes, truth = gen.trace(duration=trace_len, rate_cps=rate)
        v = codes_to_mv(codes, det)
        t0 = time.perf_counter()
        out = ch.process(v, v_max=vmax)
        dt += time.perf_counter() - t0
        amps.append(out["amplitudes"])
        n_true += len(truth["arrival_times"])
        n_rej += out["n_rejected"]
        n_trig += out["n_triggered"]
        n_sat += out["n_saturated"]
        n_samples += len(v)
    return {
        "amps": np.concatenate(amps),
        "n_true": n_true,
        "n_trig": n_trig,
        "n_rej": n_rej,
        "n_sat": n_sat,
        "seconds": dt,
        "n_samples": n_samples,
        "chain": ch,
    }


def fig5_spectrum(det):
    src = SourceConfig()
    cfg = ChainConfig(peaking_time=4e-6, flat_top=1.0e-6, tau_preamp=det.tau_preamp)
    r = collect_spectrum(det, src, cfg)

    fw, mu = fwhm_pct(r["amps"][r["amps"] > 0.75 * np.percentile(r["amps"], 99)])
    kev = 661.657 / mu

    # detector floor, same statistics, no chain at all
    gen = PulseGenerator(det=det, src=src, seed=3)
    _, floor_amps = gen.event_amplitudes(len(r["amps"]))
    fw_floor, mu_floor = fwhm_pct(
        floor_amps[floor_amps > 0.75 * np.percentile(floor_amps, 99)]
    )

    print("\n--- Fig 5: spectrum through the chain ---")
    print(f"  events generated        : {r['n_true']}")
    print(f"  triggered               : {r['n_trig']}")
    print(f"  rejected (pile-up)      : {r['n_rej']}  ({100*r['n_rej']/max(r['n_trig'],1):.1f}%)")
    print(f"  rejected (ADC clipped)  : {r['n_sat']}")
    print(f"  in final spectrum       : {len(r['amps'])}")
    print(f"  FWHM at 662 keV, chain  : {fw:6.2f} %")
    print(f"  FWHM at 662 keV, floor  : {fw_floor:6.2f} %   (detector alone)")
    deg = np.sqrt(max(fw**2 - fw_floor**2, 0.0))
    print(f"  chain's own contribution: {deg:6.2f} %   (quadrature difference)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))

    h1, e1 = np.histogram(r["amps"] * kev, bins=400, range=(0, 900))
    c1 = 0.5 * (e1[1:] + e1[:-1])
    a1.step(c1, h1, where="mid", color=GREEN, lw=0.9)
    a1.set_xlabel("energy (keV)")
    a1.set_ylabel("counts / bin")
    a1.set_title(f"Spectrum out of the DPP chain — FWHM {fw:.2f}%", loc="left")

    kev_f = 661.657 / mu_floor
    h2, e2 = np.histogram(floor_amps * kev_f, bins=400, range=(0, 900))
    a2.step(c1, h2, where="mid", color=INK, lw=0.9, label=f"detector floor {fw_floor:.2f}%")
    a2.step(c1, h1 * (h2.max() / max(h1.max(), 1)), where="mid", color=GREEN,
            lw=0.9, alpha=0.85, label=f"through chain {fw:.2f}%")
    a2.set_xlim(560, 780)
    a2.set_xlabel("energy (keV)")
    a2.set_ylabel("counts / bin (scaled)")
    a2.set_title("Photopeak: how much did the chain cost?", loc="left")
    a2.legend(frameon=False, fontsize=8)

    fig.suptitle("Figure 5 — Spectrum, and the price of processing", y=0.98, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("fig5_spectrum.png", bbox_inches="tight")
    plt.show()


# ==========================================================================
def fig7_pz_mismatch(det):
    """What happens when the pole-zero constant is set wrong.

    The cancellation is computed directly from the preamplifier
    decay time τ (Salathe & Kihm, 2016, Sec. 2.1),
    and in a digital system that value is calibrated from the detector's
    own signal rather than given (Messai et al., 2013, p. 38).
    If the value used differs from the true one, the trapezoid's flat
    top is no longer flat: it slopes, and every amplitude measured on it
    is wrong by an amount that depends on where you measure.
    """
    gen = PulseGenerator(det=det, seed=2)
    t, v = gen.single_pulse(661.657, duration=400e-6)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for frac, col, lab in [
        (0.6, COOL, "tau set 40% LOW — overshoot"),
        (1.0, GREEN, "tau correct — flat"),
        (1.6, ACCENT, "tau set 60% HIGH — undershoot"),
    ]:
        cfg = ChainConfig(peaking_time=4e-6, flat_top=3e-6,
                          tau_preamp=det.tau_preamp * frac)
        ch = DPPChain(cfg, det.sample_rate)
        s = trapezoidal_filter(v, ch.k, ch.l, ch.M) / ch.gain
        ax.plot(t * 1e6, s, color=col, lw=1.3, label=lab)

    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlim(0, 60)
    ax.set_xlabel("time (us)")
    ax.set_ylabel("shaped output (mV equiv.)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("fig7_pz_mismatch.png", dpi=300, bbox_inches="tight")
    fig.savefig("fig7_pz_mismatch.pdf", bbox_inches="tight")
    plt.show()


# ==========================================================================
if __name__ == "__main__":
    det = DetectorConfig()
    gen = PulseGenerator(det=det, seed=42)

    fig4_stages(det, gen)
    fig5_spectrum(det)
    fig7_pz_mismatch(det)

    print("\nFigures written to fig{4,5,7}_*.png  (Fig 6: run fig6_v2.py)")
