"""
phase0_iara_bench.py -- the FIRST systematic multi-method benchmark on the IARA
underwater-acoustic archive (Zenodo 10.5281/zenodo.15758636).

Honest scientific question: is the modest 5-class accuracy reported in
phase0_iara.py (~0.41) a property of the IARA task itself, or an artefact of an
impoverished front end + tiny model? To answer it we run the STANDARD digital
UATR stack (log-mel + CNN / ResNet) on the SAME recordings and the SAME
leakage-free, recording-level split, then place every method -- including the
band-limited analog PNN -- on one accuracy-vs-compute table.

Methods
  digital, full-band log-mel:  LogMel-MLP, CNN2D, ResNet-small (strong baseline)
  digital, resonator-band PSD: MLP            (from phase0_iara_results.json)
  analog  resonator-band:      PNN-PAT        (from phase0_iara_results.json)

Metric: recording-level balanced accuracy (segments aggregated per recording),
mean +/- std over seeds. Compute: MACs/inference (digital) as an energy proxy;
the PNN's energy advantage is the data-independent device-model figure.

Run:  py phase0_iara_bench.py
Out:  phase0_iara_bench_results.json, phase0_iara_bench_figure.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iara_dataset as iara
import iara_logmel as ilm
import models_bench as mb

torch.set_num_threads(16)

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
CACHE = ROOT / "iara_logmel_features.npz"
SEEDS = [0, 1, 2, 3, 4]           # 5 seeds: parity with the resonator-band channels
EPOCHS = 30
E_MAC_PJ = 3.1                    # representative INT8 digital MAC energy (pJ)


def rec_balanced_acc(prob, yte, rec_te, k):
    yn = yte.numpy()
    preds, trues = [], []
    for r in np.unique(rec_te):
        m = rec_te == r
        preds.append(int(prob[m].mean(0).argmax()))
        trues.append(int(yn[m][0]))
    preds, trues = np.array(preds), np.array(trues)
    return float(np.mean([(preds[trues == c] == c).mean()
                          for c in range(k) if (trues == c).any()]))


def train_eval(model_name, X, y, rec, k, n_mels, T, seeds):
    accs = []
    for s in seeds:
        torch.manual_seed(s)
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        rec_te = rec[te]
        m = mb.build(model_name, n_mels, T, k)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(Xtr)
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            m.train()
            for i in range(0, n, 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            prob = torch.softmax(m(Xte), dim=1).numpy()
        accs.append(rec_balanced_acc(prob, yte, rec_te, k))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "vals": [round(a, 4) for a in accs]}


def subset(X, y, rec, kk):
    if kk == 2:
        return X, (y != 0).long(), rec
    keep = (y <= (kk - 1))
    return X[keep], y[keep], rec[keep.numpy()]


def main():
    t0 = time.time()
    X, y, rec, classes = ilm.load_iara_logmel(
        DATA, str(DATA / "iara.xlsx"), CLASSES, cache=str(CACHE))
    n_mels, T = X.shape[2], X.shape[3]
    print(f"[bench] log-mel X{tuple(X.shape)}  n_rec {len(np.unique(rec))}")

    # MAC / param accounting for the digital log-mel models
    compute = {}
    for nm in ("logmel_mlp", "cnn2d", "resnet"):
        mdl = mb.build(nm, n_mels, T, 5)
        macs = mb.count_macs(mdl, (1, n_mels, T))
        compute[nm] = {"params": mb.count_params(mdl), "macs": macs,
                       "energy_pJ_per_inf": round(macs * E_MAC_PJ, 1)}
    print("[bench] compute:", json.dumps(compute))

    # digital log-mel models across granularity (checkpoint partial JSON after
    # each granularity so a killed background run never loses completed work)
    partial = ROOT / "phase0_iara_bench_partial.json"
    bench = {}
    for kk in (2, 3, 4, 5):
        Xs, ys, rs = subset(X, y, rec, kk)
        bench[kk] = {}
        for nm in ("logmel_mlp", "cnn2d", "resnet"):
            r = train_eval(nm, Xs, ys, rs, kk, n_mels, T, SEEDS)
            bench[kk][nm] = r
            print(f"  {kk}-class {nm:11s} {r['mean']:.3f}+/-{r['std']:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        partial.write_text(json.dumps({str(k): v for k, v in bench.items()}, indent=2),
                           encoding="utf-8")

    # pull the resonator-band MLP + PNN-PAT rows from the existing 1-D run
    prev = json.loads((ROOT / "phase0_iara_results.json").read_text(encoding="utf-8"))
    psd = {int(kk): v for kk, v in prev["granularity_sweep"].items()}

    res = {
        "dataset": "IARA (Zenodo 10.5281/zenodo.15758636), full archive A-H, 128 kHz hydrophone audio",
        "claim": "FIRST systematic multi-method benchmark on IARA: standard log-mel digital "
                 "stack vs band-limited analog PNN, identical recordings & recording-level split.",
        "classes_5": list(classes),
        "split": "recording-level (no segment leakage), 30% test",
        "metric": "recording-level balanced accuracy (mean +/- std over seeds)",
        "n_seeds_logmel": len(SEEDS), "n_seeds_psd": prev.get("n_seeds"),
        "epochs_logmel": EPOCHS, "logmel_shape": [int(n_mels), int(T)],
        "compute_logmel": compute,
        "results": {},
    }
    for kk in (2, 3, 4, 5):
        res["results"][str(kk)] = {
            "digital_psd_mlp": psd[kk]["digital"],          # weak feature, tiny model
            "digital_logmel_mlp": bench[kk]["logmel_mlp"],
            "digital_logmel_cnn2d": bench[kk]["cnn2d"],
            "digital_logmel_resnet": bench[kk]["resnet"],   # STRONG digital baseline
            "analog_pnn_pat": psd[kk]["pnn_pat"],           # band-limited analog
        }
    (ROOT / "phase0_iara_bench_results.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    _make_figure(res)
    print(f"\n[bench] done in {time.time()-t0:.0f}s -> phase0_iara_bench_results.json")
    # headline summary
    for kk in (2, 5):
        r = res["results"][str(kk)]
        print(f"{kk}-class: PSD-MLP {r['digital_psd_mlp']['mean']:.3f} | "
              f"logmel-ResNet {r['digital_logmel_resnet']['mean']:.3f} | "
              f"PNN-PAT {r['analog_pnn_pat']['mean']:.3f}")


def _make_figure(res):
    ks = [2, 3, 4, 5]
    methods = [("digital_logmel_resnet", "Digital ResNet (log-mel)", "#1a3e6e", "s-"),
               ("digital_logmel_cnn2d", "Digital CNN2D (log-mel)", "#2c6fbb", "^-"),
               ("digital_logmel_mlp", "Digital MLP (log-mel)", "#6fa8dc", "v-"),
               ("digital_psd_mlp", "Digital MLP (resonator PSD)", "#999999", "D--"),
               ("analog_pnn_pat", "Analog PNN-PAT (resonator)", "#cc5500", "o-")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for key, lab, col, sty in methods:
        mu = [res["results"][str(kk)][key]["mean"] for kk in ks]
        sd = [res["results"][str(kk)][key]["std"] for kk in ks]
        ax1.errorbar(ks, mu, yerr=sd, fmt=sty, color=col, capsize=2.5, label=lab, lw=1.4, ms=5)
    ax1.plot(ks, [1.0 / kk for kk in ks], ":", color="k", lw=0.9, label="chance")
    ax1.set_xticks(ks); ax1.set_xlabel("number of classes")
    ax1.set_ylabel("recording-level balanced accuracy")
    ax1.set_ylim(0, 0.95); ax1.grid(alpha=0.25); ax1.legend(fontsize=6.5, loc="upper right")
    ax1.set_title("IARA first benchmark: digital log-mel stack\nvs band-limited analog PNN", fontsize=9)

    # accuracy vs compute at 5-class
    comp = res["compute_logmel"]
    pts = [("digital_logmel_mlp", "logmel_mlp", "MLP"),
           ("digital_logmel_cnn2d", "cnn2d", "CNN2D"),
           ("digital_logmel_resnet", "resnet", "ResNet")]
    xs = [comp[c]["macs"] / 1e6 for _, c, _ in pts]
    ys = [res["results"]["5"][k]["mean"] for k, _, _ in pts]
    ax2.plot(xs, ys, "o-", color="#1a3e6e", ms=6)
    for (k, c, lab), x, yv in zip(pts, xs, ys):
        ax2.annotate(lab, (x, yv), textcoords="offset points", xytext=(5, 4), fontsize=7)
    pnn5 = res["results"]["5"]["analog_pnn_pat"]["mean"]
    ax2.axhline(pnn5, ls="--", color="#cc5500", lw=1.2)
    ax2.text(xs[-1], pnn5 + 0.005, "PNN-PAT (analog, ~94x lower energy)",
             color="#cc5500", fontsize=6.5, ha="right")
    ax2.set_xscale("log"); ax2.set_xlabel("digital MACs / inference (log)")
    ax2.set_ylabel("5-class balanced accuracy")
    ax2.set_ylim(0, 0.6); ax2.grid(alpha=0.25)
    ax2.set_title("Accuracy vs compute (5-class):\nwhere the analog point sits", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "phase0_iara_bench_figure.png", dpi=200)


if __name__ == "__main__":
    main()
