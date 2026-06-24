"""
spice_energy_update.py -- recompute the energy ledger's headline ratios and the
duty-cycle crossover using the CIRCUIT-MEASURED analog front-end power (ngspice)
in place of the original ~13 mW estimate. Reuses energy_ledger.ITEMS verbatim for
every digital line item; only the analog front end is replaced.

Revision (ARS round): the lean exemplar is the 12-mode bank (95.8 mW, 0.357
5-class) -- which Pareto-dominates the earlier 16-mode point (0.328 @ 126 mW) on
the mode sweep -- ; ratios are now reported with Monte-Carlo 95% CIs (the original
ledger carried MC, the first recompute dropped it); and a front-end sensitivity
sweep stress-tests the (fragile) edge-NPU comparison for a sub-kHz UATR band.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from energy_ledger import ITEMS  # (nominal, lo, hi) per item, mW

RNG = np.random.default_rng(20260622)
N_MC = 40000

def nom(k): return ITEMS[k][0]
def tri(k, n):
    no, lo, hi = ITEMS[k]          # ITEMS = (nominal, low, high)
    return RNG.triangular(lo, no, hi, n)

# ---- nominal digital line items (unchanged) ----
front_end = nom("adc_wideband") + nom("front_end_dsp")          # 300
common = nom("control_interface") + nom("mcu_supervision")      # 28
cont_soc = front_end + nom("classifier_soc") + nom("soc_overhead") + common   # 1378
cont_npu = front_end + nom("classifier_edge_npu") + nom("npu_overhead") + common  # 403
dc_base = nom("wakeup_detector") + common                       # 29
dc_perd = front_end + nom("classifier_soc") + nom("soc_overhead")  # 1350
pnn_perd = nom("soc_burst_log")                                 # 15
amort = nom("recal_amortised") + nom("fallback_amortised")      # 1.5

# analog front-end power per design point (circuit-measured, mW); 12-mode = lean
ANALOG = {"full_32mode_Q40": 246.3, "lean_12mode": 95.8, "original_estimate": 13.0}


def nominal(analog_mW):
    pnn_base = analog_mW + common + amort
    return {"analog_frontend_mW": analog_mW, "pnn_base_mW": round(pnn_base, 1),
            "ratio_vs_SoC": round(cont_soc / pnn_base, 1),
            "ratio_vs_edgeNPU": round(cont_npu / pnn_base, 2),
            "crossover_duty_pct": round(100 * (pnn_base - dc_base) / (dc_perd - pnn_perd), 1)}


def montecarlo(analog_mW, frac=0.18):
    """MC 95% CI on the ratios and crossover. Digital items drawn from their
    triangular ranges (ITEMS); the analog front end drawn triangular over
    +-frac (op-amp supply-current spread ~600-900 uA)."""
    n = N_MC
    fe = tri("adc_wideband", n) + tri("front_end_dsp", n)
    cm = tri("control_interface", n) + tri("mcu_supervision", n)
    soc = fe + tri("classifier_soc", n) + tri("soc_overhead", n) + cm
    npu = fe + tri("classifier_edge_npu", n) + tri("npu_overhead", n) + cm
    dcb = tri("wakeup_detector", n) + cm
    dcp = fe + tri("classifier_soc", n) + tri("soc_overhead", n)
    am = tri("recal_amortised", n) + tri("fallback_amortised", n)
    a = RNG.triangular(analog_mW * (1 - frac), analog_mW, analog_mW * (1 + frac), n)
    pnn = a + cm + am
    r_soc = soc / pnn
    r_npu = npu / pnn
    dstar = np.clip((pnn - dcb) / (dcp - tri("soc_burst_log", n)), 0, 1)
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]
    return {"ratio_vs_SoC_ci95": ci(r_soc), "ratio_vs_edgeNPU_ci95": ci(r_npu),
            "crossover_duty_pct_ci95": [round(100 * c, 1) for c in ci(dstar)]}


# front-end sensitivity: a sub-kHz UATR digital front end may be far below 300 mW;
# how does the (already thin) edge-NPU ratio move?
def frontend_sensitivity():
    out = {}
    for fe_mW in (300, 150, 50, 20):
        npu = fe_mW + nom("classifier_edge_npu") + nom("npu_overhead") + common
        out[f"frontend_{fe_mW}mW"] = {
            "continuous_edgeNPU_mW": round(npu, 1),
            "ratio_vs_edgeNPU_full32": round(npu / (ANALOG["full_32mode_Q40"] + common + amort), 2),
            "ratio_vs_edgeNPU_lean12": round(npu / (ANALOG["lean_12mode"] + common + amort), 2)}
    return out


variants = {}
for tag, a in ANALOG.items():
    v = nominal(a)
    if tag != "original_estimate":
        v.update(montecarlo(a))
    variants[tag] = v

full, lean = variants["full_32mode_Q40"], variants["lean_12mode"]
sens = frontend_sensitivity()
res = {
    "what": "Energy ledger recomputed with circuit-measured analog front-end power, MC 95% CIs",
    "digital_unchanged": {"shared_frontend_ADC_DSP_mW": front_end,
                          "continuous_SoC_mW": cont_soc, "continuous_edgeNPU_mW": cont_npu},
    "variants": variants,
    "frontend_sensitivity": sens,
    "headline": (
        f"With the circuit-measured 246 mW front end (full 32-mode Q=40 bank) the continuous-digital "
        f"advantage is {full['ratio_vs_SoC']}x vs SoC (95% CI {full['ratio_vs_SoC_ci95']}) and "
        f"{full['ratio_vs_edgeNPU']}x vs an edge NPU (CI {full['ratio_vs_edgeNPU_ci95']}); the duty-cycle "
        f"crossover is {full['crossover_duty_pct']}% (CI {full['crossover_duty_pct_ci95']}). The lean "
        f"12-mode bank (95.8 mW, 0.357 5-class -- it Pareto-dominates the 16-mode point) gives "
        f"{lean['ratio_vs_SoC']}x/{lean['ratio_vs_edgeNPU']}x, crossover {lean['crossover_duty_pct']}%. "
        f"The edge-NPU ratio is FRAGILE: it assumes a 300 mW shared front end; for a sub-kHz UATR band a "
        f"30-50 mW front end pushes the full-bank edge-NPU ratio below 1 "
        f"({sens['frontend_50mW']['ratio_vs_edgeNPU_full32']}x), so the robust claim rests on the "
        f"continuous-SoC comparison and the sub-watt-platform capability argument, not the edge-NPU number."),
}
(ROOT / "spice_energy_update_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
for tag in ("original_estimate", "full_32mode_Q40", "lean_12mode"):
    v = variants[tag]
    extra = f" CI{v.get('ratio_vs_SoC_ci95','')}" if "ratio_vs_SoC_ci95" in v else ""
    print(f"{tag:22s} analog {v['analog_frontend_mW']:6.1f}mW  base {v['pnn_base_mW']:6.1f}  "
          f"vsSoC {v['ratio_vs_SoC']:5.1f}x{extra}  vsNPU {v['ratio_vs_edgeNPU']:5.2f}x  d*={v['crossover_duty_pct']}%")
print("\nfront-end sensitivity (edge-NPU ratio):")
for k, v in sens.items():
    print(f"  {k:16s} NPU {v['continuous_edgeNPU_mW']:6.1f}mW -> full {v['ratio_vs_edgeNPU_full32']:.2f}x | lean {v['ratio_vs_edgeNPU_lean12']:.2f}x")
print("\n" + res["headline"])
