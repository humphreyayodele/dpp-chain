"""
dpp.py — The software-defined digital pulse processing chain.

This is the thesis. Everything upstream (pulsegen.py) exists to feed it;
everything downstream exists to score it.

The chain, in order:

    digitised trace
        -> pole-zero cancellation + trapezoidal shaping   (one operator)
        -> baseline restoration
        -> triggering           (on a separate FAST filter)
        -> pile-up rejection    (on the fast triggers)
        -> flat-top amplitude measurement (window anchored to the fast timing)
        -> histogram  ==>  energy spectrum

Design note on why PZ and shaping are ONE operator here
-------------------------------------------------------
Textbook block diagrams draw pole-zero cancellation as a separate stage that
undoes the preamp's exponential tail, followed by a shaper. In the recursive
digital formulation of Jordanov & Knoll (NIM A 345 (1994) 337) and Jordanov,
Knoll, Huber & Pantazis (NIM A 353 (1994) 261) they collapse into a single
recursion: the parameter M carries the pole-zero correction (NIM A 353, p. 262,
where the M stage is described as a digital pole-zero canceller) and the k, l
delays carry the trapezoid. Implementing them separately would be
slower and no more correct. This matters for the "computational cost" half of
the performance evaluation, so it is a design decision, not an accident.

Design note on the two filters (slow for energy, fast for timing)
-----------------------------------------------------------------
The slow trapezoid is good at energy and bad at timing: its rising edge is
k samples long, so a threshold crossing on it fires late by roughly
k * threshold / height, which grows with peaking time and with 1/energy.
A window anchored to that crossing slides off the flat top, bends the energy
scale, and two pulses inside one rising edge give a single trigger.
The chain therefore runs a second, short trapezoid used only for timing and
pile-up inspection, as described by Warburton, Momayezi, Hubbard-Nelson &
Skulski (2000), Appl. Radiat. Isot. 53(4-5), 913-920 (Sec. 2.2). Measured
effect in trig_check.py, fig6_v2.py and linearity.py.
Set ChainConfig(fast_trigger=False) to reproduce the original behaviour.

Design note on the fast trigger's re-arm and second-pulse detection (D1)
------------------------------------------------------------------------
Verified against ground truth in d1_diagnose.py (17 Oct 2026): a fast pulse
stays above the trigger threshold for about 2.0 us (nominal) / 1.4 us
(low-cost) at 662 keV, because the 230 ns scintillation tail follows the
1.2 us fast trapezoid. Two failures follow from triggering only on a rising
crossing with a fixed 1.2 us hold-off:
  - a second pulse that arrives while the first is still above threshold
    gets no trigger, so pile-up rejection never sees it (merged trigger);
  - on the low-cost front end the hold-off expires before the pulse falls
    back below threshold, and noise near the slow downward crossing
    re-triggers (about 0.8 per 100 pulses).
Two optional changes fix both (both off by default, so earlier figures
reproduce exactly):
  fast_rearm_sigma  : hysteresis. After a trigger, re-arm only once the fast
                      output has fallen below this lower level (e.g. 4 sigma
                      against the 6 sigma threshold). The hold-off is kept as
                      a floor.
  fast_pileup_sigma : second-rise detection. Within a pulse, if the fast
                      output has fallen at least fast_dip_sigma below its
                      running peak and then climbs more than this far above
                      its lowest point, a second pulse has started; it is
                      added as a trigger, so pile-up rejection removes both.
Retest (d1_retest.py, 17-18 Oct 2026): re-triggers 0.8 -> 0 per 100;
resolving time 2.0 / 1.4 us -> about 0.8 us on both front ends at 662 keV.
Limits (d1_blindspot.py, Cs-137 at 8 kcps, laptop, 30 Oct 2026): a pulse
arriving on the first one's rising edge or flat top makes no dip and stays
invisible (gap < ~0.8 us nominal, < ~1.0 us low-cost); a SMALLER pulse on a
larger pulse's falling edge only slows the fall and is also missed (up to
~1.0 us nominal, ~1.4 us low-cost).
Sources: the second-rise test is the principle of Mott, US Patent 7,763,859 B2
(2010), applied there to the preamplifier slope; applying it to the fast
trapezoid with separate dip and rise margins is this chain's adaptation.
The re-arm level is ordinary comparator (Schmitt-trigger) hysteresis; no
source applying it to a DPP trigger has been read. Warburton et al. (2000)
describe neither.

Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


# --------------------------------------------------------------------------
@dataclass
class ChainConfig:
    """Tunable parameters of the DPP chain.

    peaking_time and flat_top are the two knobs the performance evaluation
    sweeps. Everything else is fixed by the front end or chosen once.
    """

    # --- shaping ---
    peaking_time: float = 4e-6    # trapezoid rise time (k), s
    flat_top: float = 1.0e-6      # trapezoid flat top width, s
    tau_preamp: float = 50e-6     # preamp decay to be cancelled, s
    #   ^ In a real deployment this is MEASURED from the detector, not known.
    #     Mis-setting it slopes the flat top and leaves a residual tail; see fig7_pz_mismatch() in run_chain.py.

    # --- baseline restoration ---
    blr_window: float = 200e-6    # averaging window for baseline estimate, s
    blr_threshold_sigma: float = 3.0  # samples above this are excluded as pulses

    # --- triggering ---
    trigger_sigma: float = 6.0    # slow-filter trigger threshold (only used if fast_trigger=False)
    guard_fraction: float = 0.25  # fraction of flat top trimmed at each end

    # --- fast (timing) filter ---
    fast_trigger: bool = True     # False = original slow-filter triggering
    fast_rise: float = 0.5e-6     # fast trapezoid rise time, s (25 samples at 50 MS/s)
    fast_flat: float = 0.2e-6     # fast trapezoid flat top, s
    fast_trigger_sigma: float = 6.0  # fast threshold, in units of the fast filter's noise sigma
    fast_rearm_sigma: Optional[float] = None   # D1 hysteresis re-arm level (e.g. 4.0); None = original
    fast_pileup_sigma: Optional[float] = None  # D1 second-rise detection (e.g. 6.0); None = off
    fast_dip_sigma: float = 2.0   # D1: dip below the running peak required before a rise counts

    # --- pile-up rejection ---
    reject_pileup: bool = True


# --------------------------------------------------------------------------
# Stage 1 — pole-zero cancellation + trapezoidal shaping
# --------------------------------------------------------------------------
def _shift(x, n):
    """Delay a signal by n samples, zero-padding the front."""
    out = np.zeros_like(x)
    if n < len(x):
        out[n:] = x[: len(x) - n]
    return out


def trapezoidal_filter(v, k, l, M):
    """Jordanov-Knoll recursive trapezoidal filter with pole-zero correction.

    v : input trace (ADC codes or mV; linear either way)
    k : rise time, in samples
    l : k + flat-top width, in samples
    M : pole-zero parameter, M = 1 / (exp(T/tau) - 1), T = sampling period
        (Jordanov et al., NIM A 353 (1994) 261, Eq. 5)

    The recursion, as published in Jordanov et al., NIM A 353 (1994) 261,
    Eqs. 1-4 (the same algorithm as Jordanov & Knoll, NIM A 345 (1994) 337,
    Eqs. 28, 30, 31, written there with r folded into s):
        d[n] = v[n] - v[n-k] - v[n-l] + v[n-k-l]
        p[n] = p[n-1] + d[n]
        r[n] = p[n] + M*d[n]
        s[n] = s[n-1] + r[n]

    Both recursions are running sums, so both vectorise to np.cumsum. That
    turns an O(N) Python loop into an O(N) C loop -- same complexity, ~100x
    the speed. The algorithm is unchanged; only the implementation is.

    Note: d[n] is zero for a constant or a straight-line input, but the two
    running sums after it are not. The pole-zero stage reads a steady input
    level as a steady preamp input current, so a constant or slow signal passes
    with gain k*l, i.e. l/(M+1) of its size once divided by filter_gain()
    (about (peaking time + flat top)/tau: 10% at 4 us + 1 us). This is the
    expected behaviour of a finite-area (unipolar) shaper (Pullia, Gritti &
    Ripamonti 1997, IEEE TNS 44(3), pp. 331-333; Zeng et al. 2017, NIM A 858,
    Eqs. 1-4). Checked in ramp_test.py. The ADC offset and slow drift are
    removed by restore_baseline(), not by this filter.
    """
    v = np.asarray(v, dtype=np.float64)
    d = v - _shift(v, k) - _shift(v, l) + _shift(v, k + l)
    p = np.cumsum(d)
    s = np.cumsum(p + M * d)
    return s


def pz_parameter(tau, sample_rate):
    """M = 1/(exp(T/tau) - 1). Uses expm1 for accuracy when T << tau.

    Source: Jordanov et al., NIM A 353 (1994) 261, Eq. 5, which also gives
    M ~ tau/T - 0.5 for tau/T > 5. Jordanov & Knoll, NIM A 345 (1994) 337,
    give only the continuous-time value M = tau/T, the leading term of the
    exact form. Derivation: a sampled exponential is a^n with a = exp(-T/tau);
    its running sum is a geometric series, and the tail cancels only if
    M = a/(1 - a).

    HONEST NOTE ON THIS CHOICE: at 50 MS/s with tau = 50 us, T/tau = 4e-4 and
    the naive exp(x)-1 form agrees with expm1 to 12 significant figures. So
    this is not rescuing anything at our operating point -- it is cheap
    insurance. The cancellation only bites when T/tau falls to ~1e-8 (much
    faster sampling, or a much longer preamp decay), where exp(x)-1 keeps
    roughly half its digits. Verified, not assumed:
        naive : 2499.500033332762
        expm1 : 2499.5000333333332
    Written down because a claim like "the naive form is wrong here" would
    have been repeated in the thesis and then not survived a viva.
    """
    T = 1.0 / sample_rate
    return 1.0 / np.expm1(T / tau)


def filter_gain(k, M):
    """Flat-top height produced by a unit-amplitude exponential (preamp) step: k·(M+1).

    M + 1 = 1/(1 - exp(-T/tau)) is the factor with which Jordanov (2016,
    NIM A 805, Eqs. 14-17, p. 67) unfolds a sampled exponential into an
    impulse; a trapezoid of k-sample rise turns that into a flat top k times
    as high (own combination; derivation in Appendix A.3). This is the
    amplitude calibration, not the gain at zero frequency, which is k*l.

    Derived from the recursion and verified in gain_check.py:
    flat top / k·(M+1) = 1.0000000000000, where k·M gave 1.0004.
    Amplitude calibration that is assumed rather than measured is how
    energy scales end up wrong.
    """
    return float(k * (M + 1))


def mad_sigma(x):
    """Robust noise estimate: 1.4826 x median absolute deviation."""
    x = np.asarray(x, dtype=np.float64)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


# --------------------------------------------------------------------------
# Stage 2 — baseline restoration
# --------------------------------------------------------------------------
def restore_baseline(s, window_samples, threshold):
    """Subtract a slowly-varying baseline estimated from pulse-free samples.

    The trapezoid output should sit at zero between events, but it does not:
    the filter passes a constant offset, a drift or mains pickup at about
    l/(M+1) of its size (see trapezoidal_filter), and slow wander from parallel
    (preamp-integrated current) noise, which has the same shape as a stream of
    tiny pulses, passes too (blr_check.py, part B1; ramp_test.py). This stage
    removes them.

    The window is centred, so it uses future samples (offline only). A centred
    average follows a straight line exactly (Pullia et al. 1997, p. 335); a
    trailing window for real time would reject 50 Hz by ~30 dB instead of
    ~76 dB (trailing_window_check.py). Limitation: the gate is a fixed band
    about zero, not about the current baseline, so a drift whose shaped residue
    approaches the band is no longer fully removed (ramp_test.py, Part C2).

    So: gate the average. Samples above `threshold` are presumed to contain
    signal and are excluded. The average runs only over what is left.

    Implemented as two cumulative sums (masked signal, and the mask itself),
    which gives a gated moving average in O(N) with no loop.
    """
    s = np.asarray(s, dtype=np.float64)
    quiet = np.abs(s) < threshold

    masked = np.where(quiet, s, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(masked)))
    ccount = np.concatenate(([0.0], np.cumsum(quiet.astype(np.float64))))

    half = window_samples // 2
    n = len(s)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half, 0, n)

    total = csum[hi] - csum[lo]
    count = ccount[hi] - ccount[lo]

    # Where the window is entirely pulse (count == 0) hold the last good value
    baseline = np.divide(total, count, out=np.zeros(n), where=count > 0)
    if (count == 0).any():
        idx = np.where(count > 0, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        baseline = baseline[idx]

    return s - baseline, baseline


# --------------------------------------------------------------------------
# Stage 3 — triggering
# --------------------------------------------------------------------------
def find_triggers(s, threshold):
    """Rising-edge crossings of `threshold`. Returns sample indices.

    One trigger per crossing, not one per sample above threshold.
    """
    above = s > threshold
    rising = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    return rising


def fast_timing(v, kf, lf, M, nsig, rearm_nsig=None, pileup_nsig=None, dip_nsig=2.0):
    """Trigger and time each pulse on a short (fast) trapezoid.

    Returns (triggers, starts, f, sigma_f):
      triggers : threshold crossings of the fast output at nsig * sigma_f,
                 with a hold-off of one fast span so noise wiggles on a
                 single pulse cannot trigger twice.
      starts   : estimated pulse start for each trigger, by HALF-HEIGHT
                 timing: find the fast pulse's maximum, take the first sample
                 at or above half of it, and step back kf/2 (the fast
                 trapezoid's rising edge is kf samples long, so it passes
                 half height kf/2 after it starts). Unlike a fixed threshold,
                 the half-height point does not move with pulse size, so the
                 estimate does not walk with energy.
      f        : the fast filter output, in input-mV units (for plotting).
      sigma_f  : its robust noise estimate.

    rearm_nsig, pileup_nsig, dip_nsig : the D1 options (see the module
    docstring). With both None (the default) this is exactly the original
    function. Use fast_timing_full() to also get the number of second pulses.
    """
    trig, starts, f, sigma_f, _ = fast_timing_full(
        v, kf, lf, M, nsig, rearm_nsig, pileup_nsig, dip_nsig)
    return trig, starts, f, sigma_f


def fast_timing_full(v, kf, lf, M, nsig, rearm_nsig=None, pileup_nsig=None,
                     dip_nsig=2.0, max_region_samples=2500):
    """fast_timing() plus the number of second pulses detected (5th value).

    Second pulses are added to `triggers`; their start is approximated as the
    detection index - kf // 2. They always lie within one slow span of their
    parent, so with reject_pileup=True both are rejected and that start is
    never used for an amplitude. max_region_samples (50 us) caps the scan
    after one trigger.
    """
    f = trapezoidal_filter(v, kf, lf, M) / filter_gain(kf, M)
    f = f - np.median(f)
    sigma_f = mad_sigma(f)
    raw = find_triggers(f, nsig * sigma_f)
    n = len(f)
    holdoff = kf + lf

    trig = []
    if rearm_nsig is None:
        for t0 in raw:
            if not trig or t0 - trig[-1] > kf + lf:
                trig.append(t0)
    else:
        # hysteresis: re-arm only after the output has fallen below rearm level
        below = np.flatnonzero(f < rearm_nsig * sigma_f)
        for t0 in raw:
            if trig:
                if t0 - trig[-1] <= holdoff:
                    continue
                j = np.searchsorted(below, t0) - 1
                if j < 0 or below[j] <= trig[-1]:
                    continue
            trig.append(t0)
    trig = np.array(trig, dtype=int)

    starts = np.empty(len(trig), dtype=int)
    for i, t0 in enumerate(trig):
        lo, hi = max(t0 - kf, 0), min(t0 + lf, n)
        seg = f[lo:hi]
        half = 0.5 * seg.max()
        t50 = lo + int(np.argmax(seg >= half))
        starts[i] = t50 - kf // 2

    # second-rise detection inside each pulse region
    second = []
    if pileup_nsig is not None and len(trig):
        rise = pileup_nsig * sigma_f
        dip = dip_nsig * sigma_f
        lvl = (rearm_nsig if rearm_nsig is not None else nsig) * sigma_f
        below = np.flatnonzero(f < lvl)
        nxt = np.searchsorted(below, trig, side="right")
        for t0, j in zip(trig, nxt):
            end = below[j] if j < len(below) else n
            end = min(max(end, t0 + holdoff), t0 + max_region_samples, n)
            seg = f[t0:end]
            pk = vmin = seg[0]
            for i in range(1, len(seg)):
                x = seg[i]
                if pk - vmin > dip and x - vmin > rise:   # fell, then climbed back
                    second.append(t0 + i)
                    pk = vmin = x
                elif x > pk:                              # still rising
                    pk = vmin = x
                elif x < vmin:
                    vmin = x

    if second:
        s_idx = np.array(second, dtype=int)
        all_t = np.concatenate([trig, s_idx])
        all_s = np.concatenate([starts, s_idx - kf // 2])
        order = np.argsort(all_t, kind="stable")
        trig, starts = all_t[order], all_s[order]
    return trig, starts, f, sigma_f, len(second)


# --------------------------------------------------------------------------
# Stage 4 — pile-up rejection
# --------------------------------------------------------------------------
def reject_piled_up(triggers, k, l):
    """Drop any event with a neighbour inside one full trapezoid span.

    A trapezoid spans k + l samples. If a second event arrives inside that
    window, the flat top of the first is corrupted by the rise of the second
    and its amplitude is meaningless. It cannot be recovered by measuring
    more carefully -- the information is gone. The only honest response is to
    throw the event away and count it as dead time.

    Only as good as the triggers it is given: on slow-filter triggers, two
    pulses closer than about k samples give ONE trigger and pass unseen. On
    fast-filter triggers the pair-resolving time is about 2.0 us (nominal) /
    1.4 us (low-cost) at 662 keV, and about 0.8 us with the D1 options on
    (d1_diagnose.py, d1_retest.py). Expected rejected fraction for Poisson
    input: 1 - exp(-2 R (k + l) / f_s); the measured rejection falls short of
    it by about 2 R delta, delta the resolving time (pileup_model_check.py).

    Returns (kept_triggers, n_rejected).
    """
    triggers = np.asarray(triggers)
    if len(triggers) < 2:
        return triggers, 0

    span = k + l
    gaps_before = np.diff(triggers, prepend=triggers[0] - 10 * span)
    gaps_after = np.diff(triggers, append=triggers[-1] + 10 * span)
    clean = (gaps_before > span) & (gaps_after > span)
    return triggers[clean], int((~clean).sum())


# --------------------------------------------------------------------------
# Stage 5 — flat-top amplitude measurement
# --------------------------------------------------------------------------
def measure_amplitudes(s, starts, k, flat_top_samples, guard_fraction):
    """Average the trapezoid over the central part of its flat top.

    `starts` must mark the START of each pulse: the flat top then runs from
    start + k to start + k + flat_top. (The original chain passed slow-filter
    threshold crossings here, which fire late; see the module docstring.)

    Why average rather than take the peak: the flat top is flat, so every
    sample on it is an independent estimate of the same amplitude. Averaging
    n of them cuts the noise contribution by sqrt(n) for free. Taking the
    single maximum sample instead throws that away AND biases high, because
    the max of a noisy flat region is above its mean.

    The guard trims each end of the flat top, since the corners are rounded
    by the finite rise of the real pulse (ballistic deficit) rather than
    being the ideal sharp trapezoid the filter assumes.
    """
    guard = int(flat_top_samples * guard_fraction)
    start = k + guard
    stop = k + flat_top_samples - guard
    if stop <= start:
        start, stop = k, k + max(flat_top_samples, 1)

    amps = np.full(len(starts), np.nan)
    for i, t0 in enumerate(starts):
        a, b = t0 + start, t0 + stop
        if a >= 0 and b < len(s):
            amps[i] = s[a:b].mean()
    return amps


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------
class DPPChain:
    """Assembles the stages and reports both physics and cost metrics."""

    def __init__(self, cfg: ChainConfig, sample_rate: float):
        self.cfg = cfg
        self.fs = sample_rate
        self.k = int(round(cfg.peaking_time * sample_rate))
        self.ft = int(round(cfg.flat_top * sample_rate))
        self.l = self.k + self.ft
        self.M = pz_parameter(cfg.tau_preamp, sample_rate)
        self.gain = filter_gain(self.k, self.M)
        self.kf = int(round(cfg.fast_rise * sample_rate))
        self.lf = self.kf + int(round(cfg.fast_flat * sample_rate))

    def process(self, v, noise_sigma=None, v_max=None):
        """Run a trace through the whole chain.

        v_max : optional ADC full-scale voltage (pulsegen.adc_full_scale_mv).
                If given, events within one slow span of a clipped sample are
                rejected and counted in n_saturated.

        Returns a dict with the intermediate signals (for plotting) and the
        event list (for histogramming).
        """
        cfg = self.cfg
        v = np.asarray(v, dtype=np.float64)

        # 1 — PZ + shaping
        s_raw = trapezoidal_filter(v, self.k, self.l, self.M)

        # noise level of the SHAPED signal, estimated robustly from the data
        # itself. Median absolute deviation, not std: std is dragged upward by
        # the pulses we are trying to exclude.
        if noise_sigma is None:
            noise_sigma = mad_sigma(s_raw)

        # 2 — baseline restoration
        blr_win = int(cfg.blr_window * self.fs)
        s, baseline = restore_baseline(
            s_raw, blr_win, cfg.blr_threshold_sigma * noise_sigma
        )
        sigma = mad_sigma(s)

        # 3 — trigger (and pulse start)
        n_second = 0
        if cfg.fast_trigger:
            trig, starts, fast, sigma_f, n_second = fast_timing_full(
                v, self.kf, self.lf, self.M, cfg.fast_trigger_sigma,
                cfg.fast_rearm_sigma, cfg.fast_pileup_sigma, cfg.fast_dip_sigma)
        else:
            trig = find_triggers(s, cfg.trigger_sigma * sigma)
            starts, fast, sigma_f = trig.copy(), None, np.nan
        n_trig = len(trig)

        # 4 — pile-up rejection (on the triggers, keep starts aligned)
        n_rej = 0
        if cfg.reject_pileup and n_trig > 1:
            span = self.k + self.l
            before = np.diff(trig, prepend=trig[0] - 10 * span)
            after = np.diff(trig, append=trig[-1] + 10 * span)
            clean = (before > span) & (after > span)
            n_rej = int((~clean).sum())
            trig, starts = trig[clean], starts[clean]

        # 4b — ADC saturation: an event near a clipped sample cannot be measured
        n_sat = 0
        if v_max is not None and len(starts):
            clipped = np.concatenate(([0], np.cumsum(v >= v_max - 1e-9)))
            span = self.k + self.l
            lo = np.clip(starts - span, 0, len(v))
            hi = np.clip(starts + span, 0, len(v))
            ok = (clipped[hi] - clipped[lo]) == 0
            n_sat = int((~ok).sum())
            trig, starts = trig[ok], starts[ok]

        # 5 — measure
        amps = measure_amplitudes(s, starts, self.k, self.ft, cfg.guard_fraction)
        good = np.isfinite(amps)

        return {
            "shaped_raw": s_raw,
            "baseline": baseline,
            "shaped": s,
            "fast": fast,
            "triggers": trig[good],
            "starts": starts[good],
            "amplitudes": amps[good] / self.gain,  # back to input mV
            "n_triggered": n_trig,
            "n_second": n_second,
            "n_rejected": n_rej,
            "n_saturated": n_sat,
            "noise_sigma": sigma,
            "fast_sigma": sigma_f,
        }
