"""
energy_ledger.py -- a UNIFIED, itemised, uncertainty-propagated power ledger for
the acoustic-sensing front end, answering reviewer comment #9:

  * a single power account with EVERY line item (wideband ADC, analog front end,
    bias, readout ADC, control interface, MCU supervision, recalibration amortised,
    digital-fallback amortised) for all three strategies -- not a single-point
    device ratio;
  * Monte-Carlo uncertainty (each item drawn from a triangular distribution over
    its plausible range) propagated to total power, the duty-cycle crossover, and
    the endurance delta -> 95% CIs, not point estimates;
  * THREE-TIER reporting, explicitly separated:
      Tier 1 device upper bound   : continuous-vs-continuous ratio (the ~94x)
      Tier 2 subsystem estimate   : FAIR duty-cycled comparison at representative d
      Tier 3 mission scenario     : endurance % for real platform classes, with
                                    each class's NATIVE endurance disclosed so no
                                    one infers a 30-day mission on a 1-day AUV.

Run:  py energy_ledger.py
Outputs: energy_ledger_results.json, energy_ledger_figure.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260615)
N_MC = 40000

# ---------------------------------------------------------------------------
# Itemised power model (mW): each item = (nominal, low, high) triangular range.
# Values are device-model estimates, NOT bench measurements (stated as such).
# ---------------------------------------------------------------------------
# The digital sensing chain is decomposed into a SHARED continuous front end
# (wideband ADC + continuous spectral/feature DSP) that EVERY digital strategy
# must run for genuine always-on passive monitoring -- and that the analog PNN
# ELIMINATES by extracting the decision in the sensing physics before
# digitisation (Eq. for "digitise only the decision, not the waveform") --
# plus a SWAPPABLE classifier (a watt-class SoC, or a lean edge NPU). This makes
# the comparison fair to a reviewer who notes the 35.6 M-MAC classifier alone
# fits on a sub-100 mW NPU: the PNN's advantage comes from removing the front
# end, which no digital classifier choice avoids, not from the classifier MACs.
ITEMS = {
    # SHARED continuous digital front end (present in ALL digital strategies;
    # ELIMINATED by the analog PNN)
    "adc_wideband":      (120.0,  60.0, 220.0),   # continuous wideband (multi-channel) ADC
    "front_end_dsp":     (180.0,  70.0, 400.0),   # continuous spectral/feature DSP (FFT/mel/DEMON), real-time
    # swappable classifier
    "classifier_soc":    (900.0, 500.0, 1300.0),  # capable continuous real-time classifier on a SoC
    "classifier_edge_npu":(60.0,  15.0, 150.0),   # lean edge accelerator (Akida / Cortex-M55+Ethos-U / Loihi class)
    "soc_overhead":      (150.0,  80.0, 300.0),   # SoC housekeeping while awake (SoC strategy only)
    "npu_overhead":       (15.0,   6.0,  40.0),   # MCU+NPU housekeeping (edge-NPU strategy)
    "control_interface":  (20.0,  10.0,  40.0),   # shared bus / sensor interface
    "mcu_supervision":     (8.0,   4.0,  16.0),   # always-on microcontroller supervisor
    # duty-cycled-digital extra line item
    "wakeup_detector":     (1.0,   0.3,   3.0),   # always-on low-power event detector
    "soc_burst_log":      (15.0,   8.0,  30.0),   # SoC burst to log/act on a decision
    # analog-PNN line items (continuous analog classification; NO wideband ADC, NO front-end DSP)
    "afe_analog":          (6.0,   3.0,  12.0),   # analog front end (filter/amplify)
    "pnn_bias":            (4.0,   2.0,   9.0),   # resonator-array bias / drive
    "readout_adc_lowrate": (3.0,   1.5,   7.0),   # low-rate readout of decision scores
    "pnn_misc":            (3.0,   1.0,   6.0),   # leakage / margin
    # amortised occasional costs for the PNN path
    "recal_amortised":     (0.5,   0.1,   2.0),   # surface-window PAT update, time-averaged
    "fallback_amortised":  (1.0,   0.2,   4.0),   # rare digital-fallback episodes, avg
}


def tri(name, n):
    nom, lo, hi = ITEMS[name]
    return RNG.triangular(lo, nom, hi, size=n)


def sample_powers(n):
    """Per-draw average power (mW). The digital front end (ADC + DSP) is shared
    by the SoC and edge-NPU strategies and is what the PNN eliminates."""
    front_end = tri("adc_wideband", n) + tri("front_end_dsp", n)          # shared; PNN removes this
    common = tri("control_interface", n) + tri("mcu_supervision", n)
    # continuous digital, capable SoC classifier (the realistic watt-class always-on workload)
    cont_soc = front_end + tri("classifier_soc", n) + tri("soc_overhead", n) + common
    # continuous digital, lean EDGE-NPU classifier -- still pays the shared front end
    cont_npu = front_end + tri("classifier_edge_npu", n) + tri("npu_overhead", n) + common
    # duty-cycled digital: low-power detector base + (front end + SoC classifier) only during events
    dc_base = tri("wakeup_detector", n) + common
    dc_perd = front_end + tri("classifier_soc", n) + tri("soc_overhead", n)
    # analog PNN: NO wideband ADC, NO front-end DSP (done in physics); only AFE + bias + low-rate readout
    pnn_base = (tri("afe_analog", n) + tri("pnn_bias", n) + tri("readout_adc_lowrate", n)
                + tri("pnn_misc", n) + common + tri("recal_amortised", n) + tri("fallback_amortised", n))
    pnn_perd = tri("soc_burst_log", n)
    return cont_soc, cont_npu, dc_base, dc_perd, pnn_base, pnn_perd


def ci(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


def main():
    n = N_MC
    cont_soc, cont_npu, dc_base, dc_perd, pnn_base, pnn_perd = sample_powers(n)
    cont = cont_soc   # the realistic continuous workload (kept name for tiers below)

    # ---- shared digital front end (ADC + DSP) that the PNN eliminates ----
    front_end = tri("adc_wideband", n) + tri("front_end_dsp", n)

    # ---- Tier 1: continuous comparison against TWO digital classifiers ----
    # The reviewer's point (a 35.6 M-MAC classifier fits on a sub-100 mW NPU) is
    # answered HONESTLY: the PNN's advantage is removing the wideband ADC + DSP
    # front end, which BOTH the SoC and the edge NPU must run continuously.
    ratio_soc = cont_soc / pnn_base
    ratio_npu = cont_npu / pnn_base
    tier1 = {
        "continuous_digital_SoC_mW": [round(float(cont_soc.mean()), 1), *ci(cont_soc)],
        "continuous_digital_edgeNPU_mW": [round(float(cont_npu.mean()), 1), *ci(cont_npu)],
        "continuous_pnn_mW": [round(float(pnn_base.mean()), 1), *ci(pnn_base)],
        "shared_frontend_ADC_DSP_mW": [round(float(front_end.mean()), 1), *ci(front_end)],
        "ratio_vs_SoC_mean": round(float(ratio_soc.mean()), 1), "ratio_vs_SoC_ci95": ci(ratio_soc),
        "ratio_vs_edgeNPU_mean": round(float(ratio_npu.mean()), 1), "ratio_vs_edgeNPU_ci95": ci(ratio_npu),
        "note": ("PNN advantage is robust to the classifier choice because it removes the shared "
                 "continuous front end (ADC + DSP); even a lean edge-NPU digital path still pays it.")}

    # ---- crossover duty d* : duty-cycled digital == analog PNN ----
    # dc_base + d*dc_perd = pnn_base + d*pnn_perd  ->  d* = (pnn_base-dc_base)/(dc_perd-pnn_perd)
    den = dc_perd - pnn_perd
    dstar = np.clip((pnn_base - dc_base) / np.where(den > 1e-6, den, np.nan), 0, 1)
    dstar = dstar[np.isfinite(dstar)]
    crossover = {"d_star_mean_pct": round(float(100 * dstar.mean()), 2),
                 "d_star_ci95_pct": [round(100 * c, 2) for c in ci(dstar)]}

    # ---- Tier 2: subsystem estimate at representative duty cycles ----
    tier2 = {}
    for d in (0.01, 0.05, 0.10, 0.30):
        pdc = dc_base + d * dc_perd
        ppnn = pnn_base + d * pnn_perd
        r = pdc / ppnn
        tier2[f"d={d:.2f}"] = {
            "dutycycled_digital_mW": [round(float(pdc.mean()), 1), *ci(pdc)],
            "analog_pnn_mW": [round(float(ppnn.mean()), 1), *ci(ppnn)],
            "ratio_digital_over_pnn_mean": round(float(r.mean()), 2),
            "ratio_ci95": ci(r),
            "pnn_saves_energy": bool(r.mean() > 1.0)}

    # ---- Tier 3: endurance % for real platform classes ----
    # endurance gain = saved average power / total platform average power.
    # Saved power (PNN vs continuous-digital compute path) at the platform's
    # typical vessel-presence duty d_op. Platform total power P_total and the
    # always-on compute share are class-specific. NATIVE endurance disclosed.
    classes = {
        # name: (P_total_W mean, P_total_W lo, hi, native_endurance, d_op)
        "propulsion AUV (REMUS-100-class)": (120.0, 80.0, 180.0, "~22 h @ 3 kn", 0.10),
        "hybrid low-power AUV":             (25.0, 15.0, 45.0, "days", 0.10),
        "buoyancy glider (Slocum-class)":   (1.0, 0.5, 3.0, "weeks-months", 0.05),
    }
    tier3 = {}
    for name, (pt, plo, phi, native, d_op) in classes.items():
        P_total = RNG.triangular(plo, pt, phi, n) * 1000.0  # mW (platform total)
        p_cont_compute = cont
        p_pnn_compute = pnn_base + d_op * pnn_perd
        # the continuous-digital sensing path only FITS if it is below the platform
        # power budget; otherwise the comparison is a capability argument, not %.
        digital_feasible = float(np.mean(p_cont_compute < P_total))
        saved = np.clip(p_cont_compute - p_pnn_compute, 0, None)
        # endurance extension = saved / (new total); only meaningful where digital fits
        fits = p_cont_compute < P_total
        ext = 100.0 * saved[fits] / np.clip(P_total[fits] - saved[fits], 1e-6, None)
        entry = {"native_endurance": native, "vessel_duty_d_op": d_op,
                 "platform_total_W": round(float((P_total / 1000).mean()), 1),
                 "digital_path_feasible_frac": round(digital_feasible, 3),
                 "pnn_path_mW": round(float(p_pnn_compute.mean()), 1)}
        if digital_feasible > 0.5:
            entry["endurance_gain_pct_mean"] = round(float(ext.mean()), 2)
            entry["endurance_gain_pct_ci95"] = ci(ext)
            entry["note"] = ("gain EXTENDS the platform's native endurance by this fraction; "
                             "NOT a claim of a 30-day mission on a 1-day platform")
        else:
            entry["endurance_gain_pct_mean"] = None
            entry["note"] = ("continuous-digital sensing EXCEEDS this platform's power budget "
                             f"(~{P_total.mean()/1000:.1f} W) -> infeasible; the sub-100 mW analog "
                             "PNN is what makes always-on classification possible at all "
                             "(a CAPABILITY argument, not a % extension)")
        tier3[name] = entry

    res = {
        "what": "Unified itemised power ledger with Monte-Carlo uncertainty (device model, not bench)",
        "n_monte_carlo": n,
        "line_items_mW_nominal_lo_hi": ITEMS,
        "tier1_device_upper_bound": tier1,
        "crossover_duty": crossover,
        "tier2_subsystem_fair_dutycycled": tier2,
        "tier3_mission_endurance_by_platform": tier3,
        "honest_summary": (
            f"Tier 1: the analog PNN draws ~{tier1['continuous_pnn_mW'][0]} mW vs "
            f"~{tier1['continuous_digital_SoC_mW'][0]} mW for a continuous watt-class SoC pipeline "
            f"(ratio {tier1['ratio_vs_SoC_mean']}x) and ~{tier1['continuous_digital_edgeNPU_mW'][0]} mW for a "
            f"lean edge-NPU pipeline (ratio {tier1['ratio_vs_edgeNPU_mean']}x). The advantage survives the "
            f"edge-NPU comparison because the PNN removes the shared continuous front end "
            f"(ADC + DSP, ~{tier1['shared_frontend_ADC_DSP_mW'][0]} mW) that no digital classifier choice avoids. "
            f"Under a fair duty-cycled digital baseline the PNN saves energy only above a crossover duty "
            f"d* = {crossover['d_star_mean_pct']}% (CI {crossover['d_star_ci95_pct']}). Endurance gains are "
            f"class-dependent and EXTEND native endurance; no 30-day continuous mission is claimed for a ~1-day AUV."),
    }
    (ROOT / "energy_ledger_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _make_figure(cont_soc, cont_npu, dc_base, dc_perd, pnn_base, pnn_perd, dstar)

    print("===== UNIFIED POWER LEDGER (Monte Carlo, device model) =====")
    print(f"Tier1 PNN {tier1['continuous_pnn_mW'][0]} mW | SoC pipeline {tier1['continuous_digital_SoC_mW'][0]} mW "
          f"(ratio {tier1['ratio_vs_SoC_mean']}x) | edge-NPU pipeline {tier1['continuous_digital_edgeNPU_mW'][0]} mW "
          f"(ratio {tier1['ratio_vs_edgeNPU_mean']}x) | shared front-end {tier1['shared_frontend_ADC_DSP_mW'][0]} mW")
    print(f"crossover d* = {crossover['d_star_mean_pct']}% CI {crossover['d_star_ci95_pct']}")
    for d, v in tier2.items():
        print(f"Tier2 {d}: digital {v['dutycycled_digital_mW'][0]} mW vs PNN {v['analog_pnn_mW'][0]} mW "
              f"-> ratio {v['ratio_digital_over_pnn_mean']}x (PNN saves: {v['pnn_saves_energy']})")
    for nm, v in tier3.items():
        if v["endurance_gain_pct_mean"] is not None:
            print(f"Tier3 {nm}: +{v['endurance_gain_pct_mean']}% CI{v['endurance_gain_pct_ci95']} "
                  f"(native {v['native_endurance']}, digital feasible {v['digital_path_feasible_frac']})")
        else:
            print(f"Tier3 {nm}: digital INFEASIBLE (budget ~{v['platform_total_W']} W) -> "
                  f"PNN enables capability (native {v['native_endurance']})")


def _make_figure(cont_soc, cont_npu, dc_base, dc_perd, pnn_base, pnn_perd, dstar):
    d = np.logspace(-3, 0, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8))

    def flat(arr, color, label):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        ax1.fill_between(d, lo, hi, color=color, alpha=0.12)
        ax1.plot(d, np.full_like(d, np.median(arr)), color=color, lw=2, label=label)

    def band(base, perd, color, label):
        P = base[:, None] + perd[:, None] * d[None, :]
        lo, mid, hi = np.percentile(P, [2.5, 50, 97.5], axis=0)
        ax1.fill_between(d, lo, hi, color=color, alpha=0.15)
        ax1.plot(d, mid, color=color, lw=2, label=label)

    flat(cont_soc, "#cc3333", "Continuous digital (SoC pipeline)")
    flat(cont_npu, "#e08a1e", "Continuous digital (edge-NPU pipeline)")
    band(dc_base, dc_perd, "#888888", "Duty-cycled digital (fair)")
    band(pnn_base, pnn_perd, "#2c6fbb", "Analog PNN")
    dmid = float(np.median(dstar))
    ax1.axvline(dmid, ls="--", color="k", lw=1)
    ax1.annotate(f"$d^*\\approx{100*dmid:.1f}\\%$", (dmid, 1500),
                 textcoords="offset points", xytext=(6, 0), fontsize=8)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("vessel-presence duty cycle $d$")
    ax1.set_ylabel("average power (mW, device model)")
    ax1.set_title("Itemised power ledger with MC bands\n(95% CI shaded)", fontsize=9)
    ax1.grid(alpha=0.25, which="both"); ax1.legend(fontsize=6.5, loc="lower right")

    # right: energy-accuracy Pareto. All digital strategies run the same classifier
    # capability (the 35.6M-MAC ResNet fits on an edge NPU), so all sit at 0.460;
    # the PNN trades 0.05 accuracy for far lower power. The edge-NPU point shows
    # that even a lean classifier stays well above the PNN because of the front end.
    pts = [("Continuous digital (SoC)", float(np.median(cont_soc)), 0.460, "#cc3333"),
           ("Continuous digital (edge-NPU)", float(np.median(cont_npu)), 0.460, "#e08a1e"),
           ("Analog PNN", float(np.median(pnn_base + 0.10 * pnn_perd)), 0.403, "#2c6fbb")]
    for lab, x, yv, c in pts:
        ax2.scatter(x, yv, s=70, color=c, zorder=3)
        ax2.annotate(lab, (x, yv), textcoords="offset points", xytext=(6, 5), fontsize=6.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("average power (mW) at d = 10%")
    ax2.set_ylabel("IARA 5-class balanced accuracy")
    ax2.set_ylim(0.35, 0.5)
    ax2.set_title("Energy-accuracy Pareto (5-class, d = 10%)", fontsize=9)
    ax2.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "energy_ledger_figure.png", dpi=200)


if __name__ == "__main__":
    main()
