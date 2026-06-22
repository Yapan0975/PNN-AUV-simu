"""
iara_linear_control.py -- the control experiment the Devil's-Advocate review asked
for: a TRAINED LINEAR readout (multinomial logistic regression) on the raw 256-dim
resonator-band power spectrum, with NO resonator nonlinearity. It isolates whether
the physical-reservoir nonlinear feature map earns its keep over a trained linear
band-energy classifier, on the same leakage-free by-recording split.

Comparators already on record (5-seed, recording-level balanced accuracy):
  digital PSD-MLP (nonlinear)        5-class 0.414
  analog PNN-PAT  (reservoir+readout) 5-class 0.403   detection ROC-AUC 0.793
  fixed reservoir (random+readout)    5-class 0.378
  trivial band-energy detector (no train)        detection ROC-AUC 0.689

This script adds:
  trained LINEAR readout on band powers   5-class + detection ROC-AUC + P_d@P_fa
and P_d@P_fa=0.1 for the linear path (standard passive-sonar operating point).

Run:  py -u iara_linear_control.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import iara_dataset as iara
from phase0_iara import rec_balanced_acc, _probs

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "phase0_iara_features.npz"
SEEDS = [0, 1, 2, 3, 4]
K = 5


def train_linear(Xtr, ytr, seed, epochs=200, lr=3e-2, wd=1e-4):
    """Multinomial logistic regression = a single linear layer + softmax CE.
    No hidden layer, no nonlinearity: a trained LINEAR readout on band powers."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(Xtr.shape[1], K)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); F.cross_entropy(lin(Xtr[idx]), ytr[idx]).backward(); opt.step()
    return lin


def train_mlp(Xtr, ytr, seed, epochs=200, lr=3e-3, wd=1e-4, h=64):
    """A small DIGITAL nonlinearity on the same band powers: 1 hidden ReLU layer.
    The control the Devil's-Advocate asked for -- does the PHYSICAL reservoir beat a
    cheap digital nonlinearity, or only match it (at far lower energy)?"""
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], h), torch.nn.ReLU(),
                              torch.nn.LayerNorm(h), torch.nn.Linear(h, K))
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); F.cross_entropy(mlp(Xtr[idx]), ytr[idx]).backward(); opt.step()
    return mlp


def roc_auc_and_pd(scores, labels, pfa=0.1):
    """Manual ROC trapezoid AUC + detection prob at fixed false-alarm rate (numpy)."""
    s = np.asarray(scores); y = np.asarray(labels)
    order = np.argsort(-s); y = y[order]
    P = y.sum(); N = len(y) - P
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / max(P, 1); fpr = fp / max(N, 1)
    fpr_r = np.concatenate([[0.0], fpr]); tpr_r = np.concatenate([[0.0], tpr])
    auc = float(np.sum((fpr_r[1:] - fpr_r[:-1]) * (tpr_r[1:] + tpr_r[:-1]) / 2.0))
    pd_at = float(np.interp(pfa, fpr_r, tpr_r))   # P_d at fixed P_fa
    return round(auc, 3), round(pd_at, 3), round(float(P / len(y)), 3)


def rec_ship_scores(prob_ship, yb, rec_te):
    out = []
    for r in np.unique(rec_te):
        m = rec_te == r
        out.append((float(prob_ship[m].mean()), int(yb[m][0])))
    return out


def main():
    d = np.load(CACHE, allow_pickle=True)
    X = torch.tensor(d["X"]); y = torch.tensor(d["y"]).long(); rec = d["rec"]
    print(f"[linear-control] X {tuple(X.shape)}  classes {list(d['classes'])}", flush=True)

    five, five_mlp = [], []
    det_pts, det_pts_mlp = [], []   # pooled (score, ship) over seeds for detection
    for s in SEEDS:
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        rec_te = rec[te]
        lin = train_linear(Xtr, ytr, s)
        prob = _probs(lambda S: lin(S), Xte)
        five.append(rec_balanced_acc(prob, yte, rec_te, K))
        det_pts += rec_ship_scores(1.0 - prob[:, 0], yte.numpy(), rec_te)
        mlp = train_mlp(Xtr, ytr, s)               # digital nonlinearity on same band powers
        probm = _probs(lambda S: mlp(S), Xte)
        five_mlp.append(rec_balanced_acc(probm, yte, rec_te, K))
        det_pts_mlp += rec_ship_scores(1.0 - probm[:, 0], yte.numpy(), rec_te)
        print(f"  seed {s}: 5-class linear {five[-1]:.3f}  digital-MLP {five_mlp[-1]:.3f}", flush=True)

    five = np.array(five); five_mlp = np.array(five_mlp)
    sc = [p[0] for p in det_pts]; lb = [(0 if l == 0 else 1) for l in [p[1] for p in det_pts]]
    auc, pd10, prev = roc_auc_and_pd(sc, lb, pfa=0.1)
    scm = [p[0] for p in det_pts_mlp]; lbm = [(0 if l == 0 else 1) for l in [p[1] for p in det_pts_mlp]]
    auc_mlp, pd10_mlp, _ = roc_auc_and_pd(scm, lbm, pfa=0.1)

    out = {
        "control": "trained LINEAR readout (multinomial logistic regression) on 256-dim "
                   "resonator-band power spectrum, no resonator nonlinearity; leakage-free "
                   "by-recording split, 5 seeds",
        "linear_five_class_mean": round(float(five.mean()), 3),
        "linear_five_class_std": round(float(five.std()), 3),
        "linear_detection_roc_auc": auc, "linear_detection_pd_at_pfa_0.1": pd10,
        "digital_mlp_five_class_mean": round(float(five_mlp.mean()), 3),
        "digital_mlp_five_class_std": round(float(five_mlp.std()), 3),
        "digital_mlp_detection_roc_auc": auc_mlp, "digital_mlp_detection_pd_at_pfa_0.1": pd10_mlp,
        "prevalence_ship": prev,
        "reference": {"psd_mlp_5cls": 0.414, "pnn_pat_5cls": 0.403, "fixed_reservoir_5cls": 0.378,
                      "pnn_pat_det_auc": 0.793, "energy_det_auc": 0.689},
    }
    (ROOT / "iara_linear_control_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n[linear-control] RESULT (rec-level, 5 seeds, leakage-free)")
    print(f"  5-class:  trained-linear {out['linear_five_class_mean']}  digital-MLP {out['digital_mlp_five_class_mean']}"
          f"   (reservoir PNN-PAT 0.403, fixed-reservoir 0.378)")
    print(f"  det AUC:  trained-linear {auc}  digital-MLP {auc_mlp}   (reservoir 0.793, energy 0.689)")
    print(f"  P_d@Pfa0.1: linear {pd10}  digital-MLP {pd10_mlp}  (reservoir 0.52; prevalence {prev})")
    print("  -> wrote iara_linear_control_results.json")


if __name__ == "__main__":
    main()
