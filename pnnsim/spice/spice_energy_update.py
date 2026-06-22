"""
spice_energy_update.py -- recompute the energy ledger's headline ratios and the
duty-cycle crossover using the CIRCUIT-MEASURED analog front-end power (ngspice)
in place of the original ~13 mW estimate. Reuses energy_ledger.ITEMS verbatim for
every digital line item; only the analog front end is replaced.

Two analog variants:
  full  -- 32-mode Q=40 Tow-Thomas bank, 246 mW (circuit-measured)
  lean  -- 16-mode bank, 126 mW (within-CI accuracy per the mode sweep)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from energy_ledger import ITEMS  # nominal/lo/hi per item (mW)

def nom(k): return ITEMS[k][0]

# digital line items (unchanged)
front_end = nom("adc_wideband") + nom("front_end_dsp")          # 300
common = nom("control_interface") + nom("mcu_supervision")      # 28
cont_soc = front_end + nom("classifier_soc") + nom("soc_overhead") + common   # 1378
cont_npu = front_end + nom("classifier_edge_npu") + nom("npu_overhead") + common  # 403
dc_base = nom("wakeup_detector") + common                       # 29
dc_perd = front_end + nom("classifier_soc") + nom("soc_overhead")  # 1350
pnn_perd = nom("soc_burst_log")                                 # 15
amort = nom("recal_amortised") + nom("fallback_amortised")      # 1.5

def ledger(analog_mW, tag):
    pnn_base = analog_mW + common + amort
    r_soc = cont_soc / pnn_base
    r_npu = cont_npu / pnn_base
    dstar = (pnn_base - dc_base) / (dc_perd - pnn_perd)
    # fair duty-cycled comparison at representative duties
    duties = {}
    for d in (0.01, 0.05, 0.10, 0.30):
        pdc = dc_base + d * dc_perd
        ppnn = pnn_base + d * pnn_perd
        duties[f"d={d:.2f}"] = {"digital_mW": round(pdc, 1), "pnn_mW": round(ppnn, 1),
                                "ratio": round(pdc / ppnn, 2), "pnn_wins": bool(pdc > ppnn)}
    return {"tag": tag, "analog_frontend_mW": analog_mW,
            "pnn_base_mW": round(pnn_base, 1),
            "continuous_digital_SoC_mW": round(cont_soc, 1),
            "continuous_digital_edgeNPU_mW": round(cont_npu, 1),
            "ratio_vs_SoC": round(r_soc, 1), "ratio_vs_edgeNPU": round(r_npu, 2),
            "crossover_duty_pct": round(100 * dstar, 1), "dutycycled": duties}

full = ledger(246.0, "full_32mode_Q40")
lean = ledger(126.0, "lean_16mode")
orig = ledger(13.0, "original_estimate_for_reference")

res = {
    "what": "Energy ledger recomputed with circuit-measured analog front-end power",
    "digital_unchanged": {"shared_frontend_ADC_DSP_mW": front_end,
                          "continuous_SoC_mW": cont_soc, "continuous_edgeNPU_mW": cont_npu},
    "variants": {"full": full, "lean": lean, "original_reference": orig},
    "headline_change": (
        f"Replacing the 13 mW analog estimate with the circuit-measured 246 mW (full 32-mode "
        f"Q=40 bank) moves the continuous-digital advantage from ~27x/9x to "
        f"{full['ratio_vs_SoC']}x (vs SoC) / {full['ratio_vs_edgeNPU']}x (vs edge-NPU), and pushes "
        f"the duty-cycle crossover from ~1.5% to {full['crossover_duty_pct']}%: against a fairly "
        f"duty-cycled digital system the analog front end now wins only above "
        f"~{full['crossover_duty_pct']:.0f}% vessel-presence duty. A leaner 16-mode bank (126 mW) "
        f"recovers part of this ({lean['ratio_vs_SoC']}x/{lean['ratio_vs_edgeNPU']}x, crossover "
        f"{lean['crossover_duty_pct']}%) at within-CI accuracy."),
}
(ROOT / "spice_energy_update_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
for v in (orig, full, lean):
    print(f"{v['tag']:32s} analog {v['analog_frontend_mW']:5.0f}mW  PNN_base {v['pnn_base_mW']:6.1f}mW  "
          f"vsSoC {v['ratio_vs_SoC']:5.1f}x  vsNPU {v['ratio_vs_edgeNPU']:4.2f}x  d*={v['crossover_duty_pct']:5.1f}%")
print("\n" + res["headline_change"])
