"""
energy_dutycycle.py -- the DECISIVE fair-baseline energy experiment.

Reviewers' central attack: the ~94x power advantage is an artifact of charging
the digital baseline a CONTINUOUS 1.5 W SoC while the PNN wakes it only 1% of
the time -- no DUTY-CYCLED digital baseline is ever modeled.

This answers it head-on. For always-on passive acoustic monitoring we compare
THREE power-management strategies vs the vessel-presence duty cycle d (fraction
of time there is actually something to classify):

  1. continuous digital : ADC+CNN+SoC awake 100% of the time          (straw man)
  2. duty-cycled digital: low-power always-on WAKE-UP DETECTOR, then the SoC is
                          woken to run the full CNN only during events (fraction
                          d). Keeps FULL digital accuracy. The FAIR competitor.
  3. analog PNN         : continuous analog CLASSIFICATION at tens of mW; SoC
                          woken only to LOG a decision (~d). Costs ~0.05-0.10
                          balanced accuracy vs digital (real IARA).

Key physical distinction the PNN trades on: duty-cycled digital can only *detect*
cheaply; *continuous classification* still needs the SoC awake. The crossover is
set by  P_analog_pnn  vs  d * P_SoC_classify. We sweep d and the two load-bearing
device-model assumptions (analog-classifier power, detector power) -> crossover
RANGE, not a single point. Nothing is rigged: below the crossover, duty-cycled
digital wins on BOTH energy and accuracy and the PNN is simply not worth it.

Run:  py energy_dutycycle.py
Outputs: energy_dutycycle_results.json, energy_dutycycle_figure.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# device-model power terms (mW), consistent with phase0_acoustic.EnergyModel
P_SOC_FULL = 1502.0     # continuous digital: wideband ADC + CNN + Jetson-class SoC
P_SOC_LOG = 15.0        # SoC burst to LOG/act on a decision (no CNN) -- cheap
P_ANALOG_PNN = 16.0     # PNN continuous analog classification (device model, nominal)
P_DETECT = 1.0          # duty-cycled digital: always-on low-power wake-up detector
PNN_ACC_COST = 0.10     # honest real-data accuracy cost (IARA binary 0.65 vs 0.75)


def p_dutycycled_digital(d, p_detect=P_DETECT, p_soc=P_SOC_FULL):
    return p_detect + d * p_soc


def p_pnn(d, p_analog=P_ANALOG_PNN, p_log=P_SOC_LOG):
    return p_analog + d * p_log


def crossover_duty(p_analog=P_ANALOG_PNN, p_detect=P_DETECT,
                   p_soc=P_SOC_FULL, p_log=P_SOC_LOG):
    den = p_soc - p_log
    if den <= 0:
        return None
    return float(np.clip((p_analog - p_detect) / den, 0.0, 1.0))


def main():
    d = np.logspace(-3, 0, 400)
    pc = np.full_like(d, P_SOC_FULL)
    pd = p_dutycycled_digital(d)
    pn = p_pnn(d)
    d_star = crossover_duty()

    grid = []
    for pa in [8.0, 16.0, 32.0, 64.0, 100.0]:
        for pdet in [0.2, 1.0, 3.0, 10.0]:
            grid.append({"p_analog_mw": pa, "p_detect_mw": pdet,
                         "crossover_duty": crossover_duty(p_analog=pa, p_detect=pdet)})
    d_stars = [g["crossover_duty"] for g in grid if g["crossover_duty"] is not None]

    res = {
        "question": "Does the PNN energy advantage survive a FAIR duty-cycled digital baseline?",
        "power_terms_mw": {"P_SOC_FULL": P_SOC_FULL, "P_SOC_LOG": P_SOC_LOG,
                           "P_ANALOG_PNN": P_ANALOG_PNN, "P_DETECT": P_DETECT},
        "crossover_duty_nominal": d_star,
        "crossover_duty_range": [min(d_stars), max(d_stars)],
        "crossover_sweep": grid,
        "pnn_accuracy_cost_real_data": PNN_ACC_COST,
        "interpretation": (
            f"Under a FAIR duty-cycled digital baseline the analog PNN retains an energy "
            f"advantage only when vessel-presence duty exceeds ~{100*min(d_stars):.1f}-"
            f"{100*max(d_stars):.1f}% (device-model dependent). Below that, duty-cycled digital "
            f"wins on BOTH energy and accuracy. Above it, the PNN wins energy but still costs "
            f"~{PNN_ACC_COST:.2f} balanced accuracy. The 94x continuous-vs-continuous figure is "
            f"NOT a fair system claim; the honest claim is a regime-bounded energy-accuracy tradeoff."),
    }
    (ROOT / "energy_dutycycle_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax1.loglog(d, pc, color="#cc3333", lw=2, label="Continuous digital (straw man)")
    ax1.loglog(d, pd, color="#888888", lw=2, label="Duty-cycled digital (fair, full acc.)")
    ax1.loglog(d, pn, color="#2c6fbb", lw=2, label="Analog PNN ($-0.10$ acc.)")
    if d_star:
        yc = p_pnn(np.array([d_star]))[0]
        ax1.plot([d_star], [yc], "ko", ms=6)
        ax1.annotate(f"crossover $d^*\\approx{100*d_star:.1f}\\%$", (d_star, yc),
                     textcoords="offset points", xytext=(6, 12), fontsize=8)
    ax1.axvspan(1e-3, d_star, color="#888888", alpha=0.08)
    ax1.axvspan(d_star, 1, color="#2c6fbb", alpha=0.08)
    ax1.set_xlabel("vessel-presence duty cycle $d$")
    ax1.set_ylabel("average power (mW, device model)")
    ax1.set_title("Fair energy comparison vs operating regime", fontsize=9)
    ax1.grid(alpha=0.25, which="both"); ax1.legend(fontsize=6.5, loc="lower right")
    ax1.text(1.4e-3, 2400, "digital\nwins", fontsize=8, color="#555")
    ax1.text(0.18, 2400, "PNN wins energy\n(at $-0.10$ acc.)", fontsize=8, color="#2c6fbb")

    pas = [8, 16, 32, 64, 100]
    for pdet, mk in zip([0.2, 1.0, 3.0, 10.0], ["o", "s", "^", "d"]):
        ys = [crossover_duty(p_analog=pa, p_detect=pdet) * 100 for pa in pas]
        ax2.plot(pas, ys, marker=mk, label=f"detector {pdet} mW")
    ax2.set_xlabel("analog-PNN classifier power (mW)")
    ax2.set_ylabel("crossover duty $d^*$ (%)")
    ax2.set_title("Crossover vs device-model assumptions", fontsize=9)
    ax2.grid(alpha=0.25); ax2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / "energy_dutycycle_figure.png", dpi=200)

    print("===== FAIR DUTY-CYCLED ENERGY COMPARISON =====")
    print(f"continuous digital ........ {P_SOC_FULL:.0f} mW (flat)")
    print(f"analog PNN ................. {P_ANALOG_PNN:.0f} mW + d*{P_SOC_LOG:.0f} mW")
    print(f"duty-cycled digital ....... {P_DETECT:.0f} mW + d*{P_SOC_FULL:.0f} mW")
    print(f"crossover duty d* (nominal) {100*d_star:.2f}%")
    print(f"crossover range (sweep) ... {100*min(d_stars):.2f}% - {100*max(d_stars):.2f}%")
    print(f"PNN accuracy cost (real) .. -{PNN_ACC_COST:.2f}")
    print("\nVERDICT:")
    print(res["interpretation"])


if __name__ == "__main__":
    main()
