"""
c2_stacked_level.py — Verification item C2: the stacked preamplifier level.

Ch 3 §3.2.2 says the DC-coupled preamp output sits on a mean level
V_avg ~ R x A x tau, and that random arrivals make it fluctuate so the ADC
eventually clips. This script measures that level and compares it with
Campbell's theorem for a sum of randomly arriving pulses (shot noise):

    mean      = R * E[A]   * integral(h)       (= R * E[A] * tau exactly)
    variance  = R * E[A^2] * integral(h^2)
    n-th cumulant = R * E[A^n] * integral(h^n)  -> skewness

where h is pulsegen's tail_pulse_shape per unit step (230 ns rise, 50 us
decay) and A the step height Q/C_f. (Campbell's theorem itself is NOT yet a
verified source in this project — see literature log, Campbell / Rice 1944,
TO FIND. The formulas here are checked numerically, not cited.)

For each rate and source it reports:
  - the noise-free level rebuilt from the trace's own ground truth
    (arrival times, charges), after a 400 us warm-up: mean, sd, skewness,
    and the fraction of samples at or above the ADC full-scale input
  - Campbell's mean, sd and skewness, and the Gaussian prediction for the
    fraction above full scale
  - from the digitised trace itself (same as clip_check.py): % of samples at
    the top ADC code, and % of events with a clipped sample within 3 us of
    their arrival (i.e. clipped at their own peak)

Part 2 repeats the per-event figure with a fast event-by-event recursion
(2,000,000 events per point), because at 1-2 kcps the traces hold only a few
hundred events. It is checked against the trace result at 5-20 kcps.

Writes c2_stacked_level.csv and c2_event_level.csv.
Run:  python c2_stacked_level.py   (a few minutes)
Ayodele Verification session 1, 17 Oct 2026.
"""
import csv
import math
import numpy as np
from pulsegen import (DetectorConfig, SourceConfig, PulseGenerator,
                      adc_full_scale_mv, tail_pulse_shape)

RATES = [1000, 2000, 5000, 8000, 10000, 12000, 15000, 20000]
SOURCES = [("662 keV only", SourceConfig(photopeak_fraction=1.0)),
           ("Cs-137 spectrum", SourceConfig())]
N_TRACES = 20         # seeds 0..19; seeds 0..3 are clip_check.py's
TRACE_LEN = 12e-3     # s, as clip_check.py
WARMUP = 400e-6       # s, = 8 tau: the trace starts from an empty preamp
PEAK_WIN = 3e-6       # s, window after arrival checked for clipping
N_AMP = 200_000       # events for E[A^n]
EV_RATES = [1000, 1500, 2000, 5000, 8000, 10000, 12000, 15000, 20000]
N_EV = 2_000_000      # events per point, event-level part


def shape_integrals(det):
    """Integrals of h, h^2, h^3 per unit step (h -> 1 at the step)."""
    dt = 1 / det.sample_rate
    t = np.arange(int(10 * det.tau_preamp * det.sample_rate)) * dt
    h = tail_pulse_shape(t, det) * det.c_feedback
    return h.sum() * dt, (h**2).sum() * dt, (h**3).sum() * dt


def amplitude_moments(det, src):
    """E[A], E[A^2], E[A^3] of the step height in mV."""
    _, a = PulseGenerator(det=det, src=src, seed=999).event_amplitudes(N_AMP)
    return a.mean(), (a**2).mean(), (a**3).mean()


def noise_free_level(truth, n, det, shape):
    v = np.zeros(n)
    idx = (truth["arrival_times"] * det.sample_rate).astype(int)
    kl = len(shape)
    for i, q in zip(idx, truth["charges_c"]):
        end = min(i + kl, n)
        v[i:end] += q * shape[: end - i]
    return v * 1e3


def skew(x):
    d = x - x.mean()
    return (d**3).mean() / (d**2).mean() ** 1.5


def main():
    det = DetectorConfig()
    fs_mv = adc_full_scale_mv(det)
    top = 2**det.adc_bits - 1
    i1, i2, i3 = shape_integrals(det)
    kl = int(min(8 * det.tau_preamp, TRACE_LEN) * det.sample_rate)
    shape = tail_pulse_shape(np.arange(kl) / det.sample_rate, det)
    w0 = int(WARMUP * det.sample_rate)
    pw = int(PEAK_WIN * det.sample_rate)
    print(f"full-scale input {fs_mv:.1f} mV;  integral(h) = {i1 * 1e6:.3f} us, "
          f"integral(h^2) = {i2 * 1e6:.3f} us")

    rows = []
    for name, src in SOURCES:
        m1, m2, m3 = amplitude_moments(det, src)
        print(f"\n--- {name}: E[A] = {m1:.1f} mV, sqrt(E[A^2]) = {math.sqrt(m2):.1f} mV ---")
        print(" rate | mean: Campbell  sim |  sd: Campbell  sim | skew: Campbell  sim |"
              " >FS: Gauss%   sim%  | ADC top-code %: all  cc4 | events clipped %")
        for rate in RATES:
            mean_c = rate * m1 * i1
            sd_c = math.sqrt(rate * m2 * i2)
            sk_c = rate * m3 * i3 / (rate * m2 * i2) ** 1.5
            z = (fs_mv - mean_c) / sd_c
            p_gauss = 0.5 * math.erfc(z / math.sqrt(2))

            levels, top_all, top_cc4, n_ev, n_ev_clip = [], [], [], 0, 0
            for s in range(N_TRACES):
                gen = PulseGenerator(det=det, src=src, seed=s)
                _, codes, truth = gen.trace(TRACE_LEN, rate_cps=rate)
                is_top = codes == top
                if s < 4:
                    top_cc4.append(is_top.mean())        # clip_check.py's measure
                top_all.append(is_top[w0:].mean())
                levels.append(noise_free_level(truth, len(codes), det, shape)[w0:])
                c = np.concatenate(([0], np.cumsum(is_top)))
                idx = (truth["arrival_times"] * det.sample_rate).astype(int)
                idx = idx[(idx >= w0) & (idx + pw < len(codes))]
                n_ev += len(idx)
                n_ev_clip += int(((c[idx + pw] - c[idx]) > 0).sum())
            lv = np.concatenate(levels)
            row = dict(source=name, rate_cps=rate,
                       mean_campbell_mv=mean_c, mean_sim_mv=lv.mean(),
                       sd_campbell_mv=sd_c, sd_sim_mv=lv.std(),
                       skew_campbell=sk_c, skew_sim=skew(lv),
                       above_fs_gauss_pct=100 * p_gauss,
                       above_fs_sim_pct=100 * (lv >= fs_mv).mean(),
                       adc_top_pct=100 * np.mean(top_all),
                       adc_top_pct_clipcheck4=100 * np.mean(top_cc4),
                       events_clipped_pct=100 * n_ev_clip / max(n_ev, 1),
                       n_events=n_ev)
            rows.append(row)
            print(f"{rate:5d} | {mean_c:8.0f} {lv.mean():6.0f} | {sd_c:8.0f} {lv.std():6.0f} |"
                  f" {sk_c:8.2f} {skew(lv):5.2f} | {100 * p_gauss:9.5f} {row['above_fs_sim_pct']:7.4f} |"
                  f" {row['adc_top_pct']:7.4f} {row['adc_top_pct_clipcheck4']:7.4f} |"
                  f" {row['events_clipped_pct']:6.2f}")

    # --- part 2: event-level recursion, for rates where traces hold too few events ---
    print(f"\nEvent-level estimate ({N_EV:,} events per point): % of events whose own")
    print("peak reaches full scale. Validate against the trace column above (5-20 kcps).")
    tf, ts = det.tau_preamp, det.tau_scint
    t_pk = tf * ts / (tf - ts) * math.log(tf / ts)
    r_pk = (tf / (tf - ts)) * (math.exp(-t_pk / tf) - math.exp(-t_pk / ts))
    ev_rows = []
    for name, src in SOURCES:
        print(f"  {name}")
        for rate in EV_RATES:
            pct = event_level_clip_pct(det, src, rate, fs_mv, tf, t_pk, r_pk)
            trace_pct = next((r["events_clipped_pct"] for r in rows
                              if r["source"] == name and r["rate_cps"] == rate), None)
            ev_rows.append(dict(source=name, rate_cps=rate, n_events=N_EV,
                                events_clipped_pct_eventlevel=pct,
                                events_clipped_pct_trace=trace_pct if trace_pct is not None else ""))
            tp = f"{trace_pct:6.2f}" if trace_pct is not None else "     -"
            print(f"    {rate:6d} cps   event-level {pct:9.5f} %   trace {tp} %")

    with open("c2_stacked_level.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    with open("c2_event_level.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ev_rows[0].keys()))
        w.writeheader()
        for r in ev_rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nWrote c2_stacked_level.csv and c2_event_level.csv")


def event_level_clip_pct(det, src, rate, fs_mv, tf, t_pk, r_pk):
    """Fraction of events whose own pulse peak reaches full scale.

    Carries the preamp level from event to event with exponential decay:
        L_i = L_{i-1} * exp(-dt_i / tau) + A_i        (A = step height, mV)
    and tests  L_{i-1} * exp(-(dt_i + t_pk)/tau) + r_pk * A_i >= full scale,
    i.e. the level the new pulse lands on, decayed to the time of its peak,
    plus its own peak. Ignores electronic noise (a few mV), the rise of a
    pulse that arrived < 1.24 us earlier, and pulses that arrive during the
    new pulse's rise. Same physics as pulsegen; checked against the trace
    measurement at 5-20 kcps.
    """
    rng = np.random.default_rng(2026)
    _, a = PulseGenerator(det=det, src=src, seed=2027).event_amplitudes(N_EV)
    dt = rng.exponential(1 / rate, N_EV)
    decay = np.exp(-dt / tf).tolist()
    a = a.tolist()
    k_pk = math.exp(-t_pk / tf)
    level, n_clip = 0.0, 0
    for d, ai in zip(decay, a):
        base = level * d
        if base * k_pk + r_pk * ai >= fs_mv:
            n_clip += 1
        level = base + ai
    return 100 * n_clip / N_EV


if __name__ == "__main__":
    main()
