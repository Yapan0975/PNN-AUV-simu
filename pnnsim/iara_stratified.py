"""
iara_stratified.py -- SNR-stratified accuracy on IARA, answering the domain
reviewer's R1 ("an accuracy-vs-CPA/SNR-bin curve for at least the strong digital
ResNet and the analog PNN is a low-cost, high-value add").

Why sea state and not CPA distance: IARA's per-recording CPA metadata is sparsely
populated in our 225-recording subset -- only 94/225 carry a numeric CPA time and
68 of those sit at the 150 s sentinel, while the 45 Background recordings (no
ship) carry none -- so a CPA-distance curve is not estimable here. We instead
stratify by SEA STATE (Douglas scale 1-7), the standard, externally validated
proxy for wind-driven ambient noise (Wenz curves) and hence in-band SNR:
  calm  (sea state 1-2): quieter background -> higher SNR
  rough (sea state >=3):  louder background -> lower SNR
Both bins carry all five classes (calm 98 rec, rough 104 rec; 23 unlabelled
recordings dropped).

Protocol: for BOTH the strong digital log-mel ResNet and the analog resonator-band
PNN-PAT we run the SAME leakage-free, recording-level 30% holdout at 5 seeds (the
two caches are the identical 225 recordings, so the splits coincide -> a fair
head-to-head). Each recording's per-seed test predictions are aggregated by
majority vote into ONE recording-level prediction (no segment leakage, no
pseudo-replication), recordings are binned by sea state, and we report
recording-level balanced accuracy per bin with a recording bootstrap 95% CI.

Run:  py -u iara_stratified.py
Out:  iara_stratified_results.json, iara_stratified_figure.png
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core
import iara_dataset as iara
import iara_logmel as ilm
import models_bench as mb
from phase0_iara import _probs

torch.set_num_threads(16)

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
PSD_CACHE = ROOT / "phase0_iara_features.npz"
LOGMEL_CACHE = ROOT / "iara_logmel_features.npz"
SEEDS = [0, 1, 2, 3, 4]
K = 5
BINS = ("calm (sea state 1-2)", "rough (sea state >=3)")


def balanced_acc(preds, trues, k=K):
    return float(np.mean([(preds[trues == c] == c).mean()
                          for c in range(k) if (trues == c).any()]))


def rec2seastate():
    di = pd.read_excel(str(DATA / "iara.xlsx"), sheet_name="dataset_info")
    di.columns = [str(c).strip() for c in di.columns]
    ss = pd.to_numeric(di["Sea state"], errors="coerce")
    return {int(i): (None if np.isnan(s) else float(s))
            for i, s in zip(di["IARA ID"], ss)}


def rec_predict_rows(prob, yte, rec_te):
    """One (rec_id, pred, true) per test recording (segments aggregated)."""
    yn = yte.numpy()
    rows = []
    for r in np.unique(rec_te):
        m = rec_te == r
        rows.append((int(r), int(prob[m].mean(0).argmax()), int(yn[m][0])))
    return rows


def resnet_rows(seeds):
    X, y, rec, _ = ilm.load_iara_logmel(DATA, str(DATA / "iara.xlsx"), CLASSES,
                                        cache=str(LOGMEL_CACHE))
    nm, T = X.shape[2], X.shape[3]
    rows = []
    for s in seeds:
        torch.manual_seed(s)
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        rec_te = rec[te]
        m = mb.build("resnet", nm, T, K)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(Xtr)
        for _ in range(30):
            perm = torch.randperm(n)
            m.train()
            for i in range(0, n, 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            prob = torch.softmax(m(Xte), 1).numpy()
        rows += rec_predict_rows(prob, yte, rec_te)
        print(f"  [resnet] seed {s} done", flush=True)
    return rows


def pnn_rows(seeds):
    cfg0 = core.Config(seed=0, n_classes=K, n_modes=32, epochs=60, batch=128, lr=3e-3)
    X, y, rec, _ = iara.load_iara(cfg0, DATA, str(DATA / "iara.xlsx"), CLASSES,
                                  seg_seconds=4.0, max_per_rec=12, recs_per_class=45,
                                  seed=0, cache=str(PSD_CACHE))
    rows = []
    for s in seeds:
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        rec_te = rec[te]
        cfg = core.Config(seed=s, n_classes=K, n_modes=32, epochs=60, batch=128, lr=3e-3)
        nrng = torch.Generator().manual_seed(s + 100)
        m, _ = core.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, nrng, return_model=True)
        prob = _probs(lambda S: m.forward_eval(S, noise_rng=nrng), Xte)
        rows += rec_predict_rows(prob, yte, rec_te)
        print(f"  [pnn-pat] seed {s} done", flush=True)
    return rows


def aggregate(rows):
    """Collapse per-seed test predictions into ONE majority-vote prediction per
    unique recording (no pseudo-replication for the bootstrap)."""
    preds = defaultdict(list)
    true = {}
    for rid, p, t in rows:
        preds[rid].append(p)
        true[rid] = t
    out = {}
    for rid, ps in preds.items():
        out[rid] = (Counter(ps).most_common(1)[0][0], true[rid])
    return out  # rid -> (pred, true)


def boot_ci(p, t, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(t)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        pp, tt = p[idx], t[idx]
        if len(np.unique(tt)) < 2:
            continue
        vals.append(balanced_acc(pp, tt))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def stratify(agg, r2ss):
    out = {"overall": None}
    allp = np.array([v[0] for v in agg.values()])
    allt = np.array([v[1] for v in agg.values()])
    out["overall"] = {"n_rec": len(allt), "bal_acc": round(balanced_acc(allp, allt), 3),
                      "bal_ci95": boot_ci(allp, allt)}
    for b in BINS:
        sel = []
        for rid, (p, t) in agg.items():
            ss = r2ss.get(int(rid))
            if ss is None:
                continue
            key = BINS[0] if ss <= 2 else BINS[1]
            if key == b:
                sel.append((p, t))
        if not sel:
            out[b] = None
            continue
        p = np.array([a for a, _ in sel])
        t = np.array([c for _, c in sel])
        out[b] = {"n_rec": len(sel), "bal_acc": round(balanced_acc(p, t), 3),
                  "bal_ci95": boot_ci(p, t),
                  "per_class_n": {CLASSES[c]: int((t == c).sum()) for c in range(K)}}
    return out


def make_figure(res):
    methods = [("Digital ResNet (log-mel)", "resnet", "#1a3e6e"),
               ("Analog PNN-PAT (resonator)", "pnn_pat", "#cc5500")]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    xb = np.arange(len(BINS))
    w = 0.36
    for j, (lab, key, col) in enumerate(methods):
        mu = [res[key][b]["bal_acc"] for b in BINS]
        ci = [res[key][b]["bal_ci95"] for b in BINS]
        err = [[mu[i] - ci[i][0] for i in range(len(BINS))],
               [ci[i][1] - mu[i] for i in range(len(BINS))]]
        ax.bar(xb + (j - 0.5) * w, mu, w, yerr=err, capsize=3, color=col, label=lab, alpha=0.9)
        for i, m in enumerate(mu):
            ax.text(xb[i] + (j - 0.5) * w, m + 0.012, f"{m:.2f}", ha="center", fontsize=7)
    ax.axhline(1.0 / K, ls=":", color="k", lw=0.9)
    ax.text(1.4, 1.0 / K + 0.006, "chance", fontsize=7, color="k")
    ax.set_xticks(xb)
    ax.set_xticklabels(["calm seas\n(state 1-2, higher SNR)",
                        "rough seas\n(state >=3, lower SNR)"], fontsize=8)
    ax.set_ylabel("5-class recording-level balanced accuracy")
    ax.set_ylim(0, 0.62)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("IARA accuracy stratified by sea state (SNR proxy)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(ROOT / "iara_stratified_figure.png", dpi=200)


def main():
    t0 = time.time()
    r2ss = rec2seastate()
    print(f"[strat] sea-state map: {sum(v is not None for v in r2ss.values())} labelled", flush=True)

    print("[strat] training digital ResNet (log-mel) ...", flush=True)
    res_rows = resnet_rows(SEEDS)
    print("[strat] training analog PNN-PAT (resonator) ...", flush=True)
    pnn_rows_ = pnn_rows(SEEDS)

    res_agg = aggregate(res_rows)
    pnn_agg = aggregate(pnn_rows_)
    out = {
        "dataset": "IARA (Zenodo 10.5281/zenodo.15758636), same 225 recordings, leakage-free recording-level split",
        "stratifier": "sea state (Douglas 1-7) as wind-noise/SNR proxy; calm=1-2 (higher SNR), rough>=3 (lower SNR)",
        "why_not_cpa": "IARA per-recording CPA distance sparsely populated in this subset "
                       "(94/225 numeric, 68 at the 150 s sentinel, 0 of 45 Background) -> not estimable.",
        "n_seeds": len(SEEDS),
        "metric": "recording-level balanced accuracy, per-recording majority vote over seeds, recording bootstrap 95% CI",
        "resnet": stratify(res_agg, r2ss),
        "pnn_pat": stratify(pnn_agg, r2ss),
    }
    # gap (digital - analog) per bin
    out["gap_resnet_minus_pnn"] = {
        b: round(out["resnet"][b]["bal_acc"] - out["pnn_pat"][b]["bal_acc"], 3)
        for b in BINS}
    (ROOT / "iara_stratified_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    make_figure(out)
    print(f"[strat] done {time.time()-t0:.0f}s -> iara_stratified_results.json", flush=True)
    for b in BINS:
        print(f"  {b:24s} ResNet {out['resnet'][b]['bal_acc']:.3f}{out['resnet'][b]['bal_ci95']} | "
              f"PNN {out['pnn_pat'][b]['bal_acc']:.3f}{out['pnn_pat'][b]['bal_ci95']} | "
              f"gap {out['gap_resnet_minus_pnn'][b]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
