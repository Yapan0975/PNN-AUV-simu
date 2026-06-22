"""
iara_detection_roc.py -- ship-vs-background DETECTION ROC/PR on IARA, answering
the reviewer's "5-class accuracy is only ~0.40, what is the application value?".

The honest positioning: the low-power analog front end is a continuous PASSIVE
ACOUSTIC MONITORING / wake-up / screening component, not a full-band high-accuracy
fine-grained classifier. The right metric for that role is detection (is a vessel
present?), reported as ROC and precision-recall curves with AUC / average
precision -- the natural figures of merit for a screening detector.

We take the SAME leakage-free, recording-level split and the SAME two models as
the benchmark (strong digital log-mel ResNet; analog resonator-band PNN-PAT),
relabel to binary (ship = any of cargo/tanker/tug/special; background = no ship),
compute a per-recording detection score (mean P(ship) over the recording's
segments), pool over 5 seeds, and report ROC-AUC and average precision with a
recording bootstrap 95% CI.

Run:  py -u iara_detection_roc.py
Out:  iara_detection_roc_results.json, iara_detection_roc_figure.png
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


def rec_scores(prob_ship, yte_bin, rec_te):
    """One (detection_score, true_binary) per test recording."""
    out = []
    for r in np.unique(rec_te):
        m = rec_te == r
        out.append((float(prob_ship[m].mean()), int(yte_bin[m][0])))
    return out


def roc_pr(scores, labels):
    """Manual ROC and PR from detection scores (numpy only)."""
    s = np.asarray(scores); y = np.asarray(labels)
    order = np.argsort(-s)
    y = y[order]
    P = y.sum(); N = len(y) - P
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / max(P, 1); fpr = fp / max(N, 1)
    rec = tp / max(P, 1)
    prec = tp / np.maximum(tp + fp, 1)
    # prepend (0,0) for ROC
    fpr_r = np.concatenate([[0.0], fpr]); tpr_r = np.concatenate([[0.0], tpr])
    auc = float(np.sum((fpr_r[1:] - fpr_r[:-1]) * (tpr_r[1:] + tpr_r[:-1]) / 2.0))
    # average precision = sum of prec * delta-recall
    rec_p = np.concatenate([[0.0], rec]); prec_p = np.concatenate([[1.0], prec])
    ap = float(np.sum((rec_p[1:] - rec_p[:-1]) * prec_p[1:]))
    pd = {f"pd_at_pfa_{q}": round(float(np.interp(q, fpr_r, tpr_r)), 3) for q in (0.01, 0.05, 0.10)}
    return {"fpr": fpr_r, "tpr": tpr_r, "rec": rec_p, "prec": prec_p,
            "auc": round(auc, 3), "ap": round(ap, 3), **pd,
            "prevalence": round(float(P / len(y)), 3)}


def boot_auc(scores, labels, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores); y = np.asarray(labels); n = len(y)
    aucs, aps = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        r = roc_pr(s[idx], y[idx]); aucs.append(r["auc"]); aps.append(r["ap"])
    return ([round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)],
            [round(float(np.percentile(aps, 2.5)), 3), round(float(np.percentile(aps, 97.5)), 3)])


def resnet_detection():
    X, y, rec, _ = ilm.load_iara_logmel(DATA, str(DATA / "iara.xlsx"), CLASSES, cache=str(LOGMEL_CACHE))
    yb = (y != 0).long()
    nm, T = X.shape[2], X.shape[3]
    pts = []
    for s in SEEDS:
        torch.manual_seed(s)
        tr, te = iara.grouped_split(yb, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte = X[tr], yb[tr], X[te]
        rec_te = rec[te]
        m = mb.build("resnet", nm, T, 2)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(Xtr)
        for _ in range(30):
            perm = torch.randperm(n); m.train()
            for i in range(0, n, 128):
                idx = perm[i:i + 128]
                opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            prob_ship = torch.softmax(m(Xte), 1)[:, 1].numpy()
        pts += rec_scores(prob_ship, yb[te].numpy(), rec_te)
        print(f"  [resnet-det] seed {s} done", flush=True)
    return pts


def pnn_detection():
    cfg0 = core.Config(seed=0, n_classes=2, n_modes=32, epochs=60, batch=128, lr=3e-3)
    X, y, rec, _ = iara.load_iara(cfg0, DATA, str(DATA / "iara.xlsx"), CLASSES,
                                  seg_seconds=4.0, max_per_rec=12, recs_per_class=45,
                                  seed=0, cache=str(PSD_CACHE))
    yb = (y != 0).long()
    pts = []
    for s in SEEDS:
        tr, te = iara.grouped_split(yb, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte, yte = X[tr], yb[tr], X[te], yb[te]
        rec_te = rec[te]
        cfg = core.Config(seed=s, n_classes=2, n_modes=32, epochs=60, batch=128, lr=3e-3)
        nrng = torch.Generator().manual_seed(s + 100)
        m, _ = core.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, nrng, return_model=True)
        prob = _probs(lambda S: m.forward_eval(S, noise_rng=nrng), Xte)
        prob_ship = prob[:, 1]
        pts += rec_scores(prob_ship, yb[te].numpy(), rec_te)
        print(f"  [pnn-det] seed {s} done", flush=True)
    return pts


def energy_detector():
    """Trivial band-power detector baseline (no training): per recording, the mean
    power in the 10--1000 Hz band (NOT peak-normalised) is the detection score.
    A higher band power is taken as 'ship present'. Evaluated on the SAME 225
    recordings; deterministic, so no seeds."""
    import io
    import re
    import wave
    import zipfile
    from collections import defaultdict
    from scipy.signal import decimate
    id2cls, id2vol = iara.build_iara_label_map(str(DATA / "iara.xlsx"))
    rng = np.random.default_rng(0)
    by_vol = defaultdict(list)
    for c in CLASSES:                       # replicate load_iara's selection (45/class, seed 0)
        ids = [iid for iid, cl in id2cls.items() if cl == c]
        rng.shuffle(ids)
        for iid in ids[:45]:
            by_vol[id2vol[iid]].append((iid, 0 if c == "Background" else 1))
    pts = []
    for vol, items in by_vol.items():
        zp = DATA / f"{vol}.zip"
        if not zp.exists():
            continue
        z = zipfile.ZipFile(str(zp))
        members = {}
        for n in z.namelist():
            mm = re.search(r"-(\d+)\.wav$", n)
            if mm:
                members[int(mm.group(1))] = n
        for iid, sh in items:
            n = members.get(iid)
            if n is None:
                continue
            wf = wave.open(io.BytesIO(z.read(n)))
            sr = wf.getframerate()
            nfr = min(wf.getnframes(), int(52 * sr))
            data = np.frombuffer(wf.readframes(nfr), dtype=np.int16).astype(np.float64)
            if wf.getnchannels() > 1:
                data = data[::wf.getnchannels()]
            q = max(1, int(round(sr / 2000.0)))     # decimate to ~2 kHz band, NO peak-norm
            while q > 13:
                data = decimate(data, 4, ftype="fir"); sr //= 4; q = max(1, int(round(sr / 2000.0)))
            if q > 1:
                data = decimate(data, q, ftype="fir"); sr //= q
            pts.append((float(np.mean(data ** 2)), sh))
    print(f"  [energy-det] {len(pts)} recordings", flush=True)
    return pts


def main():
    t0 = time.time()
    print("[det] ship-vs-background detection ROC/PR on IARA (leakage-free)", flush=True)
    res_pts = resnet_detection()
    pnn_pts = pnn_detection()
    eng_pts = energy_detector()

    out = {"task": "ship (cargo/tanker/tug/special) vs background detection, IARA leakage-free, "
                   "recording-level score pooled over 5 seeds",
           "metric": "ROC-AUC and average precision (AP); recording bootstrap 95% CI"}
    curves = {}
    for name, pts in [("resnet", res_pts), ("pnn_pat", pnn_pts), ("energy", eng_pts)]:
        sc = [p[0] for p in pts]; lb = [p[1] for p in pts]
        r = roc_pr(sc, lb)
        auc_ci, ap_ci = boot_auc(sc, lb)
        out[name] = {"n_points": len(pts), "roc_auc": r["auc"], "roc_auc_ci95": auc_ci,
                     "avg_precision": r["ap"], "avg_precision_ci95": ap_ci,
                     "pd_at_pfa_0.01": r["pd_at_pfa_0.01"], "pd_at_pfa_0.05": r["pd_at_pfa_0.05"],
                     "pd_at_pfa_0.1": r["pd_at_pfa_0.1"],
                     "prevalence_ship": r["prevalence"]}
        curves[name] = r
        print(f"  {name:9s} ROC-AUC {r['auc']} {auc_ci} | AP {r['ap']} {ap_ci} (n={len(pts)})", flush=True)
    (ROOT / "iara_detection_roc_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # figure: ROC (left) + PR (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    cols = {"resnet": "#1a3e6e", "pnn_pat": "#cc5500", "energy": "#777777"}
    labs = {"resnet": "Digital ResNet (log-mel)", "pnn_pat": "Analog PNN-PAT (resonator)",
            "energy": "Band-energy detector (baseline)"}
    for name in ("resnet", "pnn_pat", "energy"):
        r = curves[name]
        ax1.plot(r["fpr"], r["tpr"], color=cols[name], lw=1.8,
                 label=f"{labs[name]} (AUC {r['auc']})")
        ax2.plot(r["rec"], r["prec"], color=cols[name], lw=1.8,
                 label=f"{labs[name]} (AP {r['ap']})")
    ax1.plot([0, 1], [0, 1], ":", color="k", lw=0.9)
    ax1.set_xlabel("false positive rate"); ax1.set_ylabel("true positive rate")
    ax1.set_title("Detection ROC (ship vs background)", fontsize=9.5)
    ax1.grid(alpha=0.25); ax1.legend(fontsize=7, loc="lower right")
    prev = curves["resnet"]["prevalence"]
    ax2.axhline(prev, ls=":", color="k", lw=0.9); ax2.text(0.02, prev + 0.01, "chance (prevalence)", fontsize=6.5)
    ax2.set_xlabel("recall"); ax2.set_ylabel("precision")
    ax2.set_title("Detection precision-recall", fontsize=9.5)
    ax2.set_ylim(0, 1.02); ax2.grid(alpha=0.25); ax2.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(ROOT / "iara_detection_roc_figure.png", dpi=200)
    print(f"[det] done {time.time()-t0:.0f}s -> iara_detection_roc_results.json", flush=True)


if __name__ == "__main__":
    main()
