"""
spice_mode_sweep.py -- the constructive half of the circuit-level study. The full
Q=40 / 32-mode Tow-Thomas bank measures 246 mW (not the 13 mW the ledger guessed),
so the honest 27x/9x energy advantage shrinks. This asks: how FEW modes does the
real bank need to keep IARA accuracy, and what is the resulting energy-optimal
analog front-end power? (The reservoir kernel-rank diagnostic already hinted the
information lives in a handful of modes.)

For N in {4,8,12,16,24,32} we keep N modes spread across the band (subsampling the
32 SPICE-measured rows), retrain the readout (15 seeds, recording-level), and price
the bank at N*3+1 op-amps. Output: accuracy-vs-power Pareto.

Run:  py spice_mode_sweep.py     (~2-4 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import core
import iara_dataset as iara
from phase0_iara_stats import balanced_acc, cluster_bootstrap_ci
from spice_iara_eval import (train_eval, subset, CLASSES, DATA, PSD_CACHE,
                             SEEDS, N_MODES, Q_NOM)

ICC, VSUP, OPS_SEC, READOUT_MW = 760e-6, 3.3, 3, 3.0
NS = [4, 8, 12, 16, 24, 32]


def power_mW(n_modes):
    return (n_modes * OPS_SEC + 1) * ICC * VSUP * 1e3 + READOUT_MW


def main():
    t0 = time.time()
    d = np.load(ROOT / "spice_reservoir.npz")
    Hsq_full = d["Hsq_spice"].astype(np.float32)
    cfg0 = core.Config(seed=0, n_classes=5, n_modes=32)
    X, y, rec, _ = iara.load_iara(cfg0, DATA, str(DATA / "iara.xlsx"), CLASSES,
                                  seg_seconds=4.0, max_per_rec=12, recs_per_class=45,
                                  seed=0, cache=str(PSD_CACHE))
    Xs, ys, rs = subset(X, y, rec, 5)
    out = []
    for n in NS:
        idx = np.linspace(0, N_MODES - 1, n).round().astype(int)
        Hsq_t = torch.from_numpy(Hsq_full[idx])
        Qv = torch.full((n,), Q_NOM)
        bals, ppt = [], []
        for s in SEEDS:
            tr, te = iara.grouped_split(ys, rs, test_frac=0.3, seed=s)
            p, t = train_eval(Xs[tr], ys[tr], Xs[te], ys[te], rs[te], Hsq_t, 5, s,
                              add_noise=False, Qvec=Qv)
            bals.append(balanced_acc(p, t, 5)); ppt.append((p, t))
        lo, hi = cluster_bootstrap_ci(ppt, 5)
        pw = power_mW(n)
        out.append({"n_modes": n, "bal_mean": round(float(np.mean(bals)), 3),
                    "bal_ci95": [lo, hi], "analog_mW": round(pw, 1),
                    "opamps": n * OPS_SEC + 1})
        print(f"  N={n:2d}: bal {out[-1]['bal_mean']:.3f}[{lo:.2f},{hi:.2f}]  "
              f"analog {pw:.0f} mW ({out[-1]['opamps']} op-amps)", flush=True)

    # pick the knee: smallest N within 0.01 of the 32-mode accuracy
    full = out[-1]["bal_mean"]
    knee = next((o for o in out if o["bal_mean"] >= full - 0.01), out[-1])
    res = {"what": "Mode-count sweep of the SPICE reservoir: accuracy vs analog power",
           "iara_5class_recording_level": True, "seeds": len(SEEDS),
           "sweep": out, "full_32mode_bal": full,
           "energy_optimal": {"n_modes": knee["n_modes"], "bal": knee["bal_mean"],
                              "analog_mW": knee["analog_mW"],
                              "vs_full_mW": out[-1]["analog_mW"]},
           "verdict": (f"{knee['n_modes']} modes hold 5-class accuracy at {knee['bal_mean']:.3f} "
                       f"(full 32-mode {full:.3f}) for {knee['analog_mW']:.0f} mW analog vs "
                       f"{out[-1]['analog_mW']:.0f} mW for the full bank -- a "
                       f"{out[-1]['analog_mW']/knee['analog_mW']:.1f}x front-end power cut at "
                       f"<=0.01 accuracy cost."),
           "wall_clock_s": round(time.time() - t0, 1)}
    (ROOT / "spice_mode_sweep_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _figure(out, knee)
    print("\n[mode-sweep] " + res["verdict"])


def _figure(out, knee):
    ns = [o["n_modes"] for o in out]
    bal = [o["bal_mean"] for o in out]
    lo = [o["bal_ci95"][0] for o in out]; hi = [o["bal_ci95"][1] for o in out]
    pw = [o["analog_mW"] for o in out]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(pw, bal, "-o", color="#2c6fbb", ms=6)
    ax.fill_between(pw, lo, hi, color="#2c6fbb", alpha=0.12)
    for o in out:
        ax.annotate(f"{o['n_modes']}", (o["analog_mW"], o["bal_mean"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.scatter([knee["analog_mW"]], [knee["bal_mean"]], s=130, facecolors="none",
               edgecolors="#cc3333", lw=2, zorder=4, label=f"energy-optimal: {knee['n_modes']} modes")
    ax.set_xlabel("analog front-end power (mW, circuit-measured)")
    ax.set_ylabel("IARA 5-class balanced accuracy")
    ax.set_title("Mode-count vs power Pareto (ngspice bank)\nlabels = number of resonator modes", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "spice_mode_sweep_figure.png", dpi=200)


if __name__ == "__main__":
    main()
