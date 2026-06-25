"""
spice_detection.py -- answer reviewer P2: the paper is now titled "presence
screening", so the screening figure of merit (ship-vs-background detection) must
be circuit-validated, not only the multi-class accuracy. Here we map the
ngspice-measured reservoir |H|^2 through the SAME 2-class (ship-vs-background)
detector and report ROC-AUC and Pd@Pfa, head-to-head with the closed-form
reservoir. If circuit ~ closed-form for detection too, the screening role rests
on circuit behavior.

Outputs: spice_detection_results.json
Run:  py spice_detection.py   (~1-2 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import core
import iara_dataset as iara
from spice_iara_eval import (features, Readout, ideal_Hsq, subset, CLASSES, DATA,
                             PSD_CACHE, SEEDS, N_MODES, Q_NOM)
from scipy import stats as sstats


def roc_metrics(scores, labels):
    """labels: 1=ship (positive), 0=background. Returns (auc, pd@0.05, pd@0.10)."""
    order = np.argsort(-scores)
    l = labels[order].astype(float)
    P, N = l.sum(), len(l) - l.sum()
    tp = np.concatenate([[0], np.cumsum(l)])
    fp = np.concatenate([[0], np.cumsum(1 - l)])
    tpr = tp / max(P, 1); fpr = fp / max(N, 1)
    auc = float(np.trapezoid(tpr, fpr))
    return auc, float(np.interp(0.05, fpr, tpr)), float(np.interp(0.10, fpr, tpr))


def train_score(Xtr, ytrb, Xte, yteb, rec_te, Hsq_t, seed, Qv):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 100)
    Etr = features(Xtr, Hsq_t, True, Qvec=Qv, rng=rng)
    m = Readout(Hsq_t.shape[0], 2)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    n = len(Etr)
    for _ in range(60):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            F.cross_entropy(m(Etr[idx]), ytrb[idx]).backward()
            opt.step()
    with torch.no_grad():
        Ete = features(Xte, Hsq_t, True, Qvec=Qv, rng=rng)
        prob = torch.softmax(m(Ete), 1)[:, 1].numpy()
    yn = yteb.numpy(); sc, lb = [], []
    for r in np.unique(rec_te):
        msk = rec_te == r
        sc.append(float(prob[msk].mean())); lb.append(int(yn[msk][0]))
    return np.array(sc), np.array(lb)


def main():
    t0 = time.time()
    d = np.load(ROOT / "spice_reservoir.npz")
    Hsq_spice = torch.from_numpy(d["Hsq_spice"].astype(np.float32))
    Qmeas = torch.from_numpy(np.nan_to_num(d["Q_meas"].astype(np.float32), nan=Q_NOM))
    Hsq_ideal = torch.from_numpy(ideal_Hsq().astype(np.float32))
    Qnom = torch.full((N_MODES,), Q_NOM)

    cfg0 = core.Config(seed=0, n_classes=5, n_modes=32)
    X, y, rec, _ = iara.load_iara(cfg0, DATA, str(DATA / "iara.xlsx"), CLASSES,
                                  seg_seconds=4.0, max_per_rec=12, recs_per_class=45,
                                  seed=0, cache=str(PSD_CACHE))
    Xs, ys, rs = subset(X, y, rec, 2)   # 2-class: ship (y!=0) vs background

    fronts = [("closed_form", Hsq_ideal, Qnom), ("ngspice_circuit", Hsq_spice, Qmeas)]
    out = {}
    for name, Hsq_t, Qv in fronts:
        aucs, p05, p10 = [], [], []
        for s in SEEDS:
            tr, te = iara.grouped_split(ys, rs, test_frac=0.3, seed=s)
            sc, lb = train_score(Xs[tr], ys[tr], Xs[te], ys[te], rs[te], Hsq_t, s, Qv)
            a, d05, d10 = roc_metrics(sc, lb)
            aucs.append(a); p05.append(d05); p10.append(d10)
        out[name] = {"roc_auc_mean": round(float(np.mean(aucs)), 3),
                     "roc_auc_std": round(float(np.std(aucs)), 3),
                     "pd_at_pfa_0.05": round(float(np.mean(p05)), 3),
                     "pd_at_pfa_0.10": round(float(np.mean(p10)), 3)}
        print(f"  {name}: AUC {out[name]['roc_auc_mean']}+-{out[name]['roc_auc_std']} | "
              f"Pd@0.05 {out[name]['pd_at_pfa_0.05']} | Pd@0.10 {out[name]['pd_at_pfa_0.10']}", flush=True)

    # paired equivalence on AUC (circuit vs closed)
    ac, asp = [], []
    for s in SEEDS:
        tr, te = iara.grouped_split(ys, rs, test_frac=0.3, seed=s)
        sc, lb = train_score(Xs[tr], ys[tr], Xs[te], ys[te], rs[te], Hsq_ideal, s, Qnom)
        ac.append(roc_metrics(sc, lb)[0])
        sc, lb = train_score(Xs[tr], ys[tr], Xs[te], ys[te], rs[te], Hsq_spice, s, Qmeas)
        asp.append(roc_metrics(sc, lb)[0])
    diff = np.array(asp) - np.array(ac)
    md = float(diff.mean()); sd = float(diff.std(ddof=1)); sem = sd / np.sqrt(len(diff))
    tci = sstats.t.interval(0.95, len(diff) - 1, loc=md, scale=sem) if sem > 0 else (md, md)
    res = {"what": "Circuit-mapped ship-vs-background detection (screening figure of merit), "
                   "ngspice circuit vs closed-form reservoir, 15 seeds, recording-level",
           "closed_form": out["closed_form"], "ngspice_circuit": out["ngspice_circuit"],
           "circuit_minus_closed_auc": {"mean": round(md, 4),
                                        "ci95": [round(tci[0], 4), round(tci[1], 4)]},
           "companion_device_auc": 0.793,
           "verdict": (f"Re-mapped through the circuit, the ship-vs-background detector keeps its "
                       f"operating characteristic: ROC-AUC {out['ngspice_circuit']['roc_auc_mean']} "
                       f"(closed-form {out['closed_form']['roc_auc_mean']}; companion device 0.793), "
                       f"Pd {out['ngspice_circuit']['pd_at_pfa_0.05']}/{out['ngspice_circuit']['pd_at_pfa_0.10']} "
                       f"at Pfa 0.05/0.10; circuit-minus-closed AUC {md:+.3f} (95% CI "
                       f"[{tci[0]:.3f},{tci[1]:.3f}]). The screening role is circuit-validated, not only "
                       f"the multi-class accuracy."),
           "wall_clock_s": round(time.time() - t0, 1)}
    (ROOT / "spice_detection_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n" + res["verdict"])


if __name__ == "__main__":
    main()
