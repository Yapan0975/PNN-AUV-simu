"""
energy_sensitivity.py -- sensitivity of the energy advantage to the ledger's
load-bearing assumptions (external review P0#5): front-end power, analog-support
power, duty cycle, and the digital baseline (SoC / edge-NPU / dedicated spectral
ASIC). Deterministic NOMINAL point estimates; the Monte-Carlo CIs for the headline
27x/9x are in the itemised ledger (energy_ledger.py / Table). The dedicated-ASIC
baseline is the lean digital front end the review and our own Devil's-Advocate
flagged as un-modelled.

Run:  py energy_sensitivity.py   ->  energy_sensitivity_results.json
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# nominal line items (mW), from energy_ledger.py ITEMS
ADC, DSP = 120.0, 180.0                      # shared wideband front end (ADC + spectral DSP)
CLS_SOC, OH_SOC = 900.0, 150.0
CLS_NPU, OH_NPU = 60.0, 15.0
COMMON = 20.0 + 8.0                          # control interface + MCU supervisor (all strategies)
WAKE = 1.0; BURST = 15.0
PNN_CONT = 6.0 + 4.0 + 3.0 + 3.0 + COMMON + 0.5 + 1.0   # afe+bias+readout+misc+common+recal+fallback ~= 45.5

def soc_total(front_end):  return front_end + CLS_SOC + OH_SOC + COMMON
def npu_total(front_end):  return front_end + CLS_NPU + OH_NPU + COMMON

def main():
    out = {"note": "nominal point estimates (mW); headline 27x/9x are Monte-Carlo means with CIs in the itemised ledger",
           "pnn_continuous_mW_nominal": round(PNN_CONT, 1)}

    # 1) front-end (ADC+DSP) power sweep -> continuous ratio vs SoC and edge-NPU
    fe_sweep = [25, 50, 100, 175, 300, 350, 600]   # 300 = nominal (120+180)
    out["front_end_sweep"] = [
        {"front_end_mW": fe,
         "ratio_vs_SoC": round(soc_total(fe) / PNN_CONT, 1),
         "ratio_vs_edge_NPU": round(npu_total(fe) / PNN_CONT, 1)}
        for fe in fe_sweep]

    # 2) dedicated spectral-ASIC digital baseline: replace the general DSP (180 mW)
    #    with a low-power spectral-feature ASIC (~30 mW) + keep wideband ADC, paired
    #    with the lean edge-NPU classifier (the leanest realistic digital front end)
    asic_fe = ADC + 30.0
    out["dedicated_ASIC_baseline"] = {
        "front_end_mW": asic_fe, "note": "ADC 120 + spectral ASIC ~30, with edge-NPU classifier",
        "total_mW": round(npu_total(asic_fe), 1),
        "ratio_vs_PNN": round(npu_total(asic_fe) / PNN_CONT, 1),
        "with_subnyquist_ADC_60mW": round(npu_total(60.0 + 30.0) / PNN_CONT, 1)}

    # 3) analog-support power sweep -> ratio at nominal front end (300 mW)
    out["analog_support_sweep"] = [
        {"pnn_support_mW": p,
         "ratio_vs_SoC": round(soc_total(300.0) / p, 1),
         "ratio_vs_edge_NPU": round(npu_total(300.0) / p, 1)}
        for p in [46, 100, 200]]

    # 4) duty-cycle: avg power of fair duty-cycled digital vs analog PNN; crossover
    dc_base = WAKE + COMMON
    dc_perd = 300.0 + CLS_SOC + OH_SOC          # front end + SoC, only during events
    pnn_perd = BURST
    def dc_avg(d):  return dc_base + d * dc_perd
    def pnn_avg(d): return PNN_CONT + d * pnn_perd
    out["duty_cycle_sweep"] = [
        {"duty_pct": d * 100,
         "dutycycled_digital_mW": round(dc_avg(d), 1),
         "analog_pnn_mW": round(pnn_avg(d), 1),
         "savings_factor": round(dc_avg(d) / pnn_avg(d), 2)}
        for d in [0.001, 0.005, 0.01, 0.015, 0.05, 0.10, 0.30]]
    # crossover d* where dc_avg == pnn_avg
    dstar = (PNN_CONT - dc_base) / (dc_perd - pnn_perd)
    out["crossover_duty_pct_nominal"] = round(dstar * 100, 2)

    (ROOT / "energy_sensitivity_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[sens] front-end sweep (ratio vs SoC / edge-NPU):")
    for r in out["front_end_sweep"]:
        print(f"   FE {r['front_end_mW']:>4} mW : {r['ratio_vs_SoC']:>5}x / {r['ratio_vs_edge_NPU']:>4}x")
    a = out["dedicated_ASIC_baseline"]
    print(f"[sens] dedicated-ASIC baseline ({a['front_end_mW']:.0f} mW FE + NPU): {a['ratio_vs_PNN']}x  "
          f"(sub-Nyquist ADC -> {a['with_subnyquist_ADC_60mW']}x)")
    print("[sens] analog-support sweep (ratio vs SoC / NPU):")
    for r in out["analog_support_sweep"]:
        print(f"   PNN {r['pnn_support_mW']:>3} mW : {r['ratio_vs_SoC']}x / {r['ratio_vs_edge_NPU']}x")
    print(f"[sens] duty-cycle crossover d* = {out['crossover_duty_pct_nominal']}%")
    print("  -> energy_sensitivity_results.json")


if __name__ == "__main__":
    main()
