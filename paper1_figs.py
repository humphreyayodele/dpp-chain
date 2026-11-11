"""
paper1_figs.py -- Figure 1 of the paper: the block diagram of the framework.

  Figure_framework_blocks   the signal source (top row) and the processing chain
                            (bottom row), with the ground truth that every
                            result is scored against. Values shown are the
                            model's defaults (pulsegen.DetectorConfig,
                            dpp.ChainConfig with the revised trigger settings),
                            read from the code, not typed in.

Run, in the folder with pulsegen.py and dpp.py:
    python paper1_figs.py
The figure is written as .png (300 dpi) and .pdf (vector).

dpp.py and pulsegen.py are imported, not edited.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pulsegen import DetectorConfig
from dpp import ChainConfig

INK, ACCENT, COOL, GREY = "#222222", "#c1440e", "#2e5c8a", "#777777"


def save(fig, name):
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {name}.png and {name}.pdf")


# --------------------------------------------------------------------------
def fig_blocks():
    det = DetectorConfig()
    cfg = ChainConfig(fast_rearm_sigma=4.0, fast_pileup_sigma=6.0)
    top = [
        ("Energy deposition", "full-energy or single\nCompton (Klein–Nishina)"),
        ("Scintillation", f"Poisson, {det.light_yield:g} ph/keV\n"
                          f"intrinsic {100*det.intrinsic_broadening:.2f} % (σ)"),
        ("Photodetection", f"binomial: collection {det.light_collection:.2f},\n"
                           f"QE {det.quantum_efficiency:.2f}; ENF {det.excess_noise_factor:.2f}"),
        ("Preamplifier", f"rise {det.tau_scint*1e9:.0f} ns, decay {det.tau_preamp*1e6:.0f} µs\n"
                         f"C_f = {det.c_feedback*1e12:.0f} pF"),
        ("Noise + mains", f"series, flicker, parallel\n{det.mains_hz:.0f} Hz, "
                          f"{det.mains_amplitude_mv:g} mV"),
        ("ADC", f"{det.sample_rate/1e6:.0f} MS/s, {det.adc_bits} bit\n"
                f"{det.adc_range_mv:.0f} mV range"),
    ]
    bottom = [
        ("Pole-zero +\ntrapezoid", f"exact M, flat top {cfg.flat_top*1e6:g} µs\n"
                                   f"peaking 1–8 µs"),
        ("Baseline\nrestorer", f"centred {cfg.blr_window*1e6:.0f} µs,\n"
                               f"gate {cfg.blr_threshold_sigma:g}σ"),
        ("Fast trigger", f"{cfg.fast_rise*1e6:g} µs rise; {cfg.fast_trigger_sigma:g}σ_f,\n"
                         f"re-arm {cfg.fast_rearm_sigma:g}σ_f"),
        ("Pile-up\nrejection", f"window k + l each side;\nsecond rise "
                               f"{cfg.fast_dip_sigma:g}σ_f / {cfg.fast_pileup_sigma:g}σ_f"),
        ("Saturation\nrejection", "clipped sample\nwithin one span"),
        ("Amplitude", "flat-top average\n→ spectrum"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    w, h, x0, dx = 1.72, 1.05, 0.12, 1.98

    def row(items, y, colour):
        for i, (title, body) in enumerate(items):
            x = x0 + i * dx
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc="white", ec=colour, lw=1.2))
            ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=7.6,
                    fontweight="bold", color=colour)
            ax.text(x + w / 2, y + 0.1, body, ha="center", va="bottom", fontsize=6.4, color=INK)
            if i < len(items) - 1:
                ax.annotate("", xy=(x + dx, y + h / 2), xytext=(x + w, y + h / 2),
                            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9))

    row(top, 2.75, COOL)
    row(bottom, 0.55, ACCENT)
    ax.text(x0, 4.05, "Signal source (pulsegen.py): known energy and arrival time of every event",
            fontsize=8, color=COOL, va="center")
    ax.text(x0, 0.25, "Processing chain (dpp.py)", fontsize=8, color=ACCENT, va="center")
    # ADC output -> chain input, routed through the gap between the rows
    xa, xb, ymid = x0 + 5 * dx + w / 2, x0 + w / 2, 2.15
    ax.plot([xa, xa, xb], [2.75, ymid, ymid], color=GREY, lw=0.9)
    ax.annotate("", xy=(xb, 0.55 + h), xytext=(xb, ymid),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9))
    ax.text((xa + xb) / 2, ymid + 0.08, "digitised trace", fontsize=7, color=GREY,
            ha="center", va="bottom")
    # ground truth -> scoring
    ax.annotate("", xy=(x0 + 5 * dx + w + 0.06, 0.55 + h / 2), xytext=(x0 + 5 * dx + w + 0.06, 2.75 + h / 2),
                arrowprops=dict(arrowstyle="-|>", color=COOL, lw=0.9, ls="--"))
    ax.text(x0 + 5 * dx + w + 0.12, 2.15, "ground truth\n→ scoring", fontsize=7, color=COOL, va="center")
    save(fig, "Figure_framework_blocks")


if __name__ == "__main__":
    fig_blocks()
