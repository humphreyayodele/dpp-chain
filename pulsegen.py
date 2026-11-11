"""
pulsegen.py — Physics-based synthetic pulse generator for a NaI(Tl) + PMT +
charge-sensitive preamplifier front end.

Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter

M_E_C2 = 510.998950
Q_E = 1.602176634e-19


@dataclass
class DetectorConfig:
    # LIGHT YIELD — conservative baseline vs. current literature.
    # Textbook/older assumption: ~38-40 ph/keV (this model's original value).
    # Luan, N.T., Kim, H.J., Lee, H.S. et al. (2024), arXiv:2404.18551v1,
    #   measured 35 NaI(Tl) samples with the single-photoelectron technique;
    #   59.4 +/- 5.9 ph/keV is their HIGHEST value (a small bare sample;
    #   Table 4, p. 9). Small bare samples gave 50-60 ph/keV; a bare 2" crystal
    #   49.5, encapsulated 32.4. Typically quoted value ~40 ph/keV (pp. 1-2).
    # Adopted here as more representative of current crystal quality. This
    # has two consequences handled explicitly below: intrinsic_broadening is
    # recalibrated, and c_feedback is increased to preserve ADC headroom.
    light_yield: float = 59.4
    tau_scint: float = 230e-9
    light_collection: float = 0.70

    # CALIBRATION, recomputed for the light yield above:
    #   N_pe(662 keV) = 662 * 59.4 * 0.70 * 0.25 = 6881
    #   sigma_stat = sqrt(F/N_pe) = sqrt(1.15/6881) = 1.29% -> FWHM = 3.04%
    #   published FWHM, small NaI(Tl)+PMT = 6.5% -> sigma_total = 2.76%
    #   required extra (quadrature) = sqrt(2.76^2 - 1.29^2) = 2.44%
    # (Under the old 38 ph/keV assumption N_pe=4402, the statistical FWHM was
    # LARGER (3.81% vs 3.05%), so a smaller extra term, 2.24%, reached the same
    # 6.5% target.) The extra term is the INTRINSIC SCINTILLATOR BROADENING:
    # it corresponds to R_s = 5.7% FWHM of Dorenbos et al. (1995, IEEE TNS
    # 42(6), Table II, p. 2199), which combines non-proportionality, crystal
    # inhomogeneity and light-transfer fluctuations -- not non-proportionality
    # alone (R_np = 4.5%).
    intrinsic_broadening: float = 0.0244

    quantum_efficiency: float = 0.25
    pmt_gain: float = 1.0e5
    excess_noise_factor: float = 1.15

    # FEEDBACK CAPACITANCE — increased from 56 pF to 100 pF in response to
    # the light-yield update. Higher light yield means more charge per keV
    # reaching the preamp; at the old 56 pF this pushed ADC headroom down to
    # ~1070 keV -- below the K-40 line (1461 keV) this thesis is motivated by.
    # 100 pF (a common commercial value) restores headroom to ~1910 keV: the
    # energy whose mean pulse peak reaches the 3099.8 mV full scale; the peak
    # is 97.5% of the charge step because the 50 us decay starts during the
    # 230 ns rise (c3_headroom.py). N_pe(662keV)=6881, V=N_pe*G*q_e/Cf.
    #
    # NOTE: this headroom is for ONE pulse. At rate, 50 us tails stack and the
    # preamp level averages ~ rate x tau x pulse height, fluctuating about that
    # mean, so clipping has no sharp onset. Fraction of 662 keV events whose own
    # peak reaches full scale: ~0.02% at 2 kcps, 0.95% at 8, 1.8% at 10, 12% at
    # 20 kcps; full Cs-137 spectrum 0.06% at 10 and 0.7% at 20 kcps
    # (c2_stacked_level.py, 2e6 events per point). Treat ~10 kcps as this front
    # end's count-rate ceiling. (clip_check.py's earlier figures, 4 traces per
    # point, are superseded.)
    c_feedback: float = 100.0e-12
    tau_preamp: float = 50e-6
    noise_rms_mv: float = 2.0
    flicker_amplitude_mv: float = 1.0
    parallel_noise_rms_mv: float = 12.0

    # MAINS PICKUP — ASSUMED values, not sourced: a 50 Hz sinusoid (Nigerian
    # grid frequency) of 4 mV amplitude. Previously 200 Hz, which had no
    # stated justification. The trapezoid passes a slow sine at about
    # l/(M+1) of its size, so 4 mV leaves ~0.4 mV after shaping at 50 Hz as at
    # 200 Hz (blr_check.py, B1; ramp_test.py). The baseline restorer removes it.
    mains_hz: float = 50.0
    mains_amplitude_mv: float = 4.0

    sample_rate: float = 50e6
    adc_bits: int = 14
    adc_range_mv: float = 3300.0
    adc_offset_mv: float = 200.0


@dataclass
class SourceConfig:
    name: str = "Cs-137"
    gamma_energy: float = 661.657
    photopeak_fraction: float = 0.30
    activity_cps: float = 2000.0


def compton_edge(E0):
    a = E0 / M_E_C2
    return E0 * (2.0 * a) / (1.0 + 2.0 * a)


def _kn(T, E0):
    a = E0 / M_E_C2
    s = np.clip(T / E0, 0.0, 1.0 - 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        kn = 2.0 + s**2 / (a**2 * (1 - s) ** 2) + (s / (1 - s)) * (s - 2.0 / a)
    kn = np.nan_to_num(kn, nan=0.0, posinf=0.0, neginf=0.0)
    return np.where(T <= compton_edge(E0), np.clip(kn, 0, None), 0.0)


def sample_deposited_energy(n, src, rng):
    E0 = src.gamma_energy
    pp = rng.random(n) < src.photopeak_fraction
    e = np.empty(n)
    e[pp] = E0
    nc = int((~pp).sum())
    if nc:
        edge = compton_edge(E0)
        g = np.linspace(1e-3, edge * 0.9999, 4000)
        cdf = np.cumsum(_kn(g, E0))
        cdf /= cdf[-1]
        e[~pp] = np.interp(rng.random(nc), cdf, g)
    return e


def energy_to_charge(e, det, rng):
    e = np.atleast_1d(np.asarray(e, float))
    nph = rng.poisson(np.clip(det.light_yield * e, 0, None))
    npe = rng.binomial(nph, det.light_collection * det.quantum_efficiency)
    mean = det.pmt_gain * npe
    ev = max(det.excess_noise_factor - 1.0, 0.0)
    sig = det.pmt_gain * np.sqrt(ev * np.maximum(npe, 1))
    ne = rng.normal(mean, sig)
    ne *= rng.normal(1.0, det.intrinsic_broadening, size=ne.shape)
    return np.clip(ne, 0, None) * Q_E


def tail_pulse_shape(t, det):
    tf, ts = det.tau_preamp, det.tau_scint
    out = np.zeros_like(t, float)
    m = t >= 0
    out[m] = (1.0 / det.c_feedback) * (tf / (tf - ts)) * (
        np.exp(-t[m] / tf) - np.exp(-t[m] / ts)
    )
    return out


def _flicker(n, amp, rng):
    w = rng.normal(0, 1, n)
    sp = np.fft.rfft(w)
    f = np.arange(sp.size)
    f[0] = 1
    sp = sp / np.sqrt(f)
    o = np.fft.irfft(sp, n)
    s = o.std()
    return o * (amp / s) if s > 0 else o


def _parallel(n, rms, tau, fs, rng):
    a = np.exp(-1.0 / (fs * tau))
    z = rng.normal(0, 1, n)
    return lfilter([rms * np.sqrt(1 - a**2)], [1.0, -a], z)


def digitise(v_mv, det):
    nl = 2**det.adc_bits
    lsb = det.adc_range_mv / nl
    c = np.floor((v_mv + det.adc_offset_mv) / lsb)
    return np.clip(c, 0, nl - 1).astype(np.int32)


def codes_to_mv(codes, det):
    lsb = det.adc_range_mv / (2**det.adc_bits)
    return codes * lsb - det.adc_offset_mv


def adc_full_scale_mv(det):
    """Input voltage of the highest ADC code; samples at this value are clipped."""
    return codes_to_mv(2**det.adc_bits - 1, det)


class PulseGenerator:
    def __init__(self, det=None, src=None, seed=None):
        self.det = det or DetectorConfig()
        self.src = src or SourceConfig()
        self.rng = np.random.default_rng(seed)

    def single_pulse(self, e, duration=400e-6):
        det = self.det
        n = int(duration * det.sample_rate)
        t = np.arange(n) / det.sample_rate
        q = energy_to_charge([e], det, self.rng)[0]
        return t, q * tail_pulse_shape(t, det) * 1e3

    def trace(self, duration=2e-3, rate_cps=None):
        det = self.det
        rate = self.src.activity_cps if rate_cps is None else rate_cps
        n = int(duration * det.sample_rate)
        t = np.arange(n) / det.sample_rate
        ne = self.rng.poisson(rate * duration)
        arr = np.sort(self.rng.random(ne) * duration)
        en = sample_deposited_energy(ne, self.src, self.rng)
        ch = energy_to_charge(en, det, self.rng)
        kl = int(min(8 * det.tau_preamp, duration) * det.sample_rate)
        tk = np.arange(kl) / det.sample_rate
        shape = tail_pulse_shape(tk, det)
        v = np.zeros(n)
        idx = (arr * det.sample_rate).astype(int)
        for i, q in zip(idx, ch):
            end = min(i + kl, n)
            v[i:end] += q * shape[: end - i]
        v *= 1e3
        v += self.rng.normal(0, det.noise_rms_mv, n)
        v += _flicker(n, det.flicker_amplitude_mv, self.rng)
        v += _parallel(n, det.parallel_noise_rms_mv, det.tau_preamp, det.sample_rate, self.rng)
        ph = self.rng.random() * 2 * np.pi
        v += det.mains_amplitude_mv * np.sin(2 * np.pi * det.mains_hz * t + ph)
        return t, digitise(v, det), {"arrival_times": arr, "energies_kev": en, "charges_c": ch}

    def event_amplitudes(self, n):
        e = sample_deposited_energy(n, self.src, self.rng)
        q = energy_to_charge(e, self.det, self.rng)
        return e, q / self.det.c_feedback * 1e3
