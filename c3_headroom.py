"""
c3_headroom.py — Verification item C3: ADC headroom of the modelled front end.

Ch 3 §3.2.2 (inherited from the proposal and pulsegen.py's comments) says the
100 pF feedback capacitor gives ADC headroom to "approximately 1981 keV", and
that 56 pF would have given "approximately 1110 keV". This script recomputes
both from pulsegen.py itself.

Headroom here = the energy whose MEAN single pulse, on a quiet baseline at
0 mV, just reaches the ADC's full-scale input. Single pulse only: at rate the
stacked tails raise the baseline (Ch 3 §3.2.2, count-rate ceiling).

Three numbers decide it:
  1. full-scale input voltage   adc_full_scale_mv(det)
  2. mean step height at 662 keV   N_pe * G * q_e / C_f
  3. peak / step ratio of tail_pulse_shape (230 ns rise, 50 us decay)

Run:  python c3_headroom.py      (pulsegen.py in the same folder; < 10 s)

Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke
Ayodele Verification session 1, 16 Oct 2026.
"""
import numpy as np
from pulsegen import (DetectorConfig, PulseGenerator, adc_full_scale_mv,
                      tail_pulse_shape, Q_E)

E_REF = 661.657   # keV, Cs-137
N_SAMPLED = 2000  # single_pulse calls, to check the analytic mean


def mean_step_mv(det, e_kev):
    """Mean step height Q/C_f in mV: no Poisson/binomial/gain fluctuations."""
    n_pe = e_kev * det.light_yield * det.light_collection * det.quantum_efficiency
    return n_pe * det.pmt_gain * Q_E / det.c_feedback * 1e3


def peak_ratio(det):
    """Peak of tail_pulse_shape / (1/C_f): analytic and on the 50 MS/s grid."""
    tf, ts = det.tau_preamp, det.tau_scint
    t_pk = tf * ts / (tf - ts) * np.log(tf / ts)
    analytic = (tf / (tf - ts)) * (np.exp(-t_pk / tf) - np.exp(-t_pk / ts))
    t = np.arange(int(20e-6 * det.sample_rate)) / det.sample_rate
    sampled = tail_pulse_shape(t, det).max() * det.c_feedback
    return analytic, sampled, t_pk


def report(det, label):
    fs = adc_full_scale_mv(det)
    step = mean_step_mv(det, E_REF)
    r_an, r_s, t_pk = peak_ratio(det)
    peak = step * r_s
    e_peak = E_REF * fs / peak                  # peak-based, offset subtracted
    e_step = E_REF * fs / step                  # step-based, offset subtracted
    e_naive = E_REF * det.adc_range_mv / step   # full 3300 mV range, step-based
    print(f"--- {label}: C_f = {det.c_feedback * 1e12:.0f} pF ---")
    print(f"  full-scale input          : {fs:9.3f} mV")
    print(f"  mean step at 662 keV      : {step:9.2f} mV")
    print(f"  peak/step (analytic)      : {r_an:9.5f}   peak at {t_pk * 1e6:.3f} us")
    print(f"  peak/step (50 MS/s grid)  : {r_s:9.5f}")
    print(f"  mean peak at 662 keV      : {peak:9.2f} mV")
    print(f"  headroom, peak-based      : {e_peak:7.0f} keV   <- what the ADC sees")
    print(f"  headroom, step-based      : {e_step:7.0f} keV   (ignores the 230 ns rise)")
    print(f"  headroom, 3300 mV / step  : {e_naive:7.0f} keV   (ignores the 200 mV offset too)")
    return fs, peak


def main():
    fs, peak = report(DetectorConfig(), "current model")
    print()
    report(DetectorConfig(c_feedback=56e-12), "previous value")

    # check the analytic mean against single_pulse() itself
    gen = PulseGenerator(det=DetectorConfig(), seed=7)
    peaks = np.array([gen.single_pulse(E_REF, duration=20e-6)[1].max()
                      for _ in range(N_SAMPLED)])
    print(f"\nsingle_pulse(661.657) x {N_SAMPLED}: mean peak {peaks.mean():.1f} mV, "
          f"sd {peaks.std():.1f} mV ({100 * peaks.std() / peaks.mean():.2f} %), "
          f"max {peaks.max():.1f} mV")
    print(f"  analytic mean peak {peak:.1f} mV; full scale {fs:.1f} mV")


if __name__ == "__main__":
    main()
