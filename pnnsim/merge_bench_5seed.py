"""
merge_bench_5seed.py -- finish the 5-seed IARA benchmark robustly.

The full phase0_iara_bench.py 5-seed re-run is long enough to be killed by the
background-task time limit during the last (5-class) granularity. This script
computes ONLY the missing 5-class log-mel rows at 5 seeds (fast, ~7 min), reuses
the already-completed 2/3/4-class log-mel rows from phase0_iara_bench_partial.json,
pulls the resonator-band PSD-MLP and PNN-PAT rows from phase0_iara_results.json,
and assembles the final phase0_iara_bench_results.json + figure in the same schema.

Run:  py merge_bench_5seed.py
"""
from __future__ import annotations
import json
from pathlib import Path

import phase0_iara_bench as B
import iara_logmel as ilm
import models_bench as mb

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")

X, y, rec, classes = ilm.load_iara_logmel(DATA, str(DATA / "iara.xlsx"), CLASSES,
                                          cache=str(B.CACHE))
n_mels, T = X.shape[2], X.shape[3]
print(f"[merge] log-mel X{tuple(X.shape)}  seeds {B.SEEDS}", flush=True)

# compute accounting
compute = {}
for nm in ("logmel_mlp", "cnn2d", "resnet"):
    mdl = mb.build(nm, n_mels, T, 5)
    compute[nm] = {"params": mb.count_params(mdl), "macs": mb.count_macs(mdl, (1, n_mels, T)),
                   "energy_pJ_per_inf": round(mb.count_macs(mdl, (1, n_mels, T)) * B.E_MAC_PJ, 1)}

# 5-class log-mel rows at 5 seeds (the missing piece)
Xs, ys, rs = B.subset(X, y, rec, 5)
logmel5 = {}
for nm in ("logmel_mlp", "cnn2d", "resnet"):
    logmel5[nm] = B.train_eval(nm, Xs, ys, rs, 5, n_mels, T, B.SEEDS)
    print(f"  5-class {nm:11s} {logmel5[nm]['mean']:.3f}+/-{logmel5[nm]['std']:.3f}", flush=True)

# 2/3/4-class log-mel from the partial checkpoint (already 5-seed)
partial = json.loads((ROOT / "phase0_iara_bench_partial.json").read_text(encoding="utf-8"))
# resonator-band PSD-MLP + PNN-PAT from the 5-seed phase0_iara run
prev = json.loads((ROOT / "phase0_iara_results.json").read_text(encoding="utf-8"))
psd = {int(kk): v for kk, v in prev["granularity_sweep"].items()}

logmel_by_kk = {2: partial["2"], 3: partial["3"], 4: partial["4"], 5: logmel5}

res = {
    "dataset": "IARA (Zenodo 10.5281/zenodo.15758636), full archive A-H, 128 kHz hydrophone audio",
    "claim": "First leakage-free, multi-method, energy-aware benchmark on IARA (5-seed parity).",
    "classes_5": list(classes),
    "split": "recording-level (no segment leakage), 30% test",
    "n_seeds_logmel": len(B.SEEDS), "n_seeds_psd": prev.get("n_seeds"),
    "epochs_logmel": B.EPOCHS, "logmel_shape": [int(n_mels), int(T)],
    "compute_logmel": compute,
    "results": {},
}
for kk in (2, 3, 4, 5):
    lm = logmel_by_kk[kk]
    res["results"][str(kk)] = {
        "digital_psd_mlp": psd[kk]["digital"],
        "digital_logmel_mlp": lm["logmel_mlp"],
        "digital_logmel_cnn2d": lm["cnn2d"],
        "digital_logmel_resnet": lm["resnet"],
        "analog_pnn_pat": psd[kk]["pnn_pat"],
    }
(ROOT / "phase0_iara_bench_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
B._make_figure(res)
print("[merge] wrote phase0_iara_bench_results.json (5-seed) + figure", flush=True)
for kk in (2, 5):
    r = res["results"][str(kk)]
    print(f"{kk}-class: psd-mlp {r['digital_psd_mlp']['mean']:.3f} | logmel-mlp {r['digital_logmel_mlp']['mean']:.3f} "
          f"| cnn2d {r['digital_logmel_cnn2d']['mean']:.3f} | resnet {r['digital_logmel_resnet']['mean']:.3f} "
          f"| pnn {r['analog_pnn_pat']['mean']:.3f}")
