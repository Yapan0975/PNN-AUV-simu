"""
iara_leakage_test.py -- the DECISIVE experiment after discovering that the IARA
descriptor paper (IEEE Access 2025, DOI 10.1109/ACCESS.2025.3585467) already
reports a multi-method baseline (Random Forest / MLP / CNN; MLP best at
balanced accuracy 67.48 +/- 1.24%, 5x2 cross-validation).

Their 0.675 (MLP) vs our 0.41-0.45 (recording-level, leakage-free) gap must be
explained. The leading hypothesis: standard 5x2 CV in UATR splits at the
SEGMENT/window level, so windows from the SAME recording (same ship, same pass)
appear in both train and test -> recording-level leakage -> inflated accuracy.

This script tests that hypothesis DIRECTLY on our cached IARA features by
comparing, for the SAME models and data, two splits:
  (A) recording-level (our honest, leakage-free)  -> grouped_split
  (B) segment-level    (random, LEAKY, mimics naive 5x2 CV)

If (B) >> (A), the gap is largely a leakage artifact and our leakage-free number
is the honest correction. If (B) ~ (A), leakage is not the explanation and the
gap is data quantity / task difference (we use 225 of 1825 recordings).

Run:  py iara_leakage_test.py
Out:  iara_leakage_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import core
import iara_dataset as iara

torch.set_num_threads(16)

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
PSD_CACHE = ROOT / "phase0_iara_features.npz"
LOGMEL_CACHE = ROOT / "iara_logmel_features.npz"
SEEDS = [0, 1, 2, 3, 4]


def seg_split(n, test_frac, seed):
    """Naive SEGMENT-level random split (LEAKY: ignores recording grouping)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ncut = int(round((1 - test_frac) * n))
    tr = np.zeros(n, bool); te = np.zeros(n, bool)
    tr[idx[:ncut]] = True; te[idx[ncut:]] = True
    return tr, te


def seg_balanced_acc(pred, y, k):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(k) if (y == c).any()]))


def rec_balanced_acc(prob, y, rec, k):
    yn = y.numpy(); preds, trues = [], []
    for r in np.unique(rec):
        m = rec == r
        preds.append(int(prob[m].mean(0).argmax())); trues.append(int(yn[m][0]))
    preds, trues = np.array(preds), np.array(trues)
    return float(np.mean([(preds[trues == c] == c).mean() for c in range(k) if (trues == c).any()]))


def run_psd_mlp(X, y, rec, k):
    """Digital MLP on the PSD feature (the closest analogue to IARA's best model,
    an MLP), under recording-level vs segment-level splits."""
    out = {"recording_level": [], "segment_level": []}
    seg_out = {"recording_level": [], "segment_level": []}
    for s in SEEDS:
        cfg = core.Config(seed=s, n_classes=k, n_modes=32, epochs=60, batch=128, lr=3e-3)
        # recording-level (honest)
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        m = _train_mlp(cfg, X[tr], y[tr])
        with torch.no_grad():
            prob = torch.softmax(m(X[te]), 1).numpy()
        out["recording_level"].append(rec_balanced_acc(prob, y[te], rec[te], k))
        seg_out["recording_level"].append(seg_balanced_acc(prob.argmax(1), y[te].numpy(), k))
        # segment-level (leaky)
        trs, tes = seg_split(len(X), 0.3, s)
        trs = torch.from_numpy(trs); tes = torch.from_numpy(tes)
        m2 = _train_mlp(cfg, X[trs], y[trs])
        with torch.no_grad():
            prob2 = torch.softmax(m2(X[tes]), 1).numpy()
        seg_out["segment_level"].append(seg_balanced_acc(prob2.argmax(1), y[tes].numpy(), k))
    return {
        "recording_level_segacc": _ms(seg_out["recording_level"]),
        "segment_level_segacc": _ms(seg_out["segment_level"]),
        "recording_level_recacc": _ms(out["recording_level"]),
    }


def _train_mlp(cfg, Xtr, ytr):
    torch.manual_seed(cfg.seed)
    m = core.DigitalBaseline(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
    return m


def run_logmel(model_name, k, seeds=(0, 1, 2)):
    """Strong digital log-mel model under recording-level vs segment-level."""
    import iara_logmel as ilm
    import models_bench as mb
    X, y, rec, _ = ilm.load_iara_logmel(DATA, str(DATA / "iara.xlsx"), CLASSES, cache=str(LOGMEL_CACHE))
    nm, T = X.shape[2], X.shape[3]
    seg_rec, seg_seg = [], []
    for s in seeds:
        # recording-level
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        m = _train_cnn(mb, model_name, nm, T, k, X[tr], y[tr], s)
        with torch.no_grad():
            p = torch.softmax(m(X[te]), 1).numpy()
        seg_rec.append(seg_balanced_acc(p.argmax(1), y[te].numpy(), k))
        # segment-level (leaky)
        trs, tes = seg_split(len(X), 0.3, s)
        trs = torch.from_numpy(trs); tes = torch.from_numpy(tes)
        m2 = _train_cnn(mb, model_name, nm, T, k, X[trs], y[trs], s)
        with torch.no_grad():
            p2 = torch.softmax(m2(X[tes]), 1).numpy()
        seg_seg.append(seg_balanced_acc(p2.argmax(1), y[tes].numpy(), k))
    return {"recording_level_segacc": _ms(seg_rec), "segment_level_segacc": _ms(seg_seg)}


def _train_cnn(mb, name, nm, T, k, Xtr, ytr, seed):
    torch.manual_seed(seed)
    m = mb.build(name, nm, T, k)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(30):
        perm = torch.randperm(len(Xtr)); m.train()
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
    m.eval(); return m


def _ms(v):
    return {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3),
            "vals": [round(x, 3) for x in v]}


def main():
    t0 = time.time()
    k = len(CLASSES)
    cfg0 = core.Config(seed=0, n_classes=k, n_modes=32, epochs=60, batch=128, lr=3e-3)
    X, y, rec, _ = iara.load_iara(cfg0, DATA, str(DATA / "iara.xlsx"), CLASSES,
                                  seg_seconds=4.0, max_per_rec=12, recs_per_class=45,
                                  seed=0, cache=str(PSD_CACHE))
    print(f"[leak] PSD X{tuple(X.shape)} n_rec {len(np.unique(rec))}", flush=True)

    res = {
        "question": "Is the gap between IARA-paper MLP (0.675, 5x2 CV) and our recording-level "
                    "0.41-0.45 explained by segment-level leakage in naive CV?",
        "iara_paper_baseline": {"models": ["RandomForest", "MLP", "CNN"],
                                "best_MLP_balanced_acc": "0.6748 +/- 0.0124", "cv": "5x2",
                                "doi": "10.1109/ACCESS.2025.3585467"},
        "our_protocol": "225 recordings (45/class) of 1825; 4 s windows; 5 classes",
        "note_segacc": "segacc = SEGMENT-level balanced accuracy (not aggregated per recording), "
                       "so it is directly comparable to a 5x2-CV segment-level number.",
    }
    print("[leak] PSD-MLP (matches IARA's best model = MLP) ...", flush=True)
    res["psd_mlp_5class"] = run_psd_mlp(X, y, rec, k)
    print(f"  rec-level segacc {res['psd_mlp_5class']['recording_level_segacc']['mean']:.3f} | "
          f"seg-level segacc {res['psd_mlp_5class']['segment_level_segacc']['mean']:.3f}", flush=True)

    for nm in ("logmel_mlp", "cnn2d", "resnet"):
        print(f"[leak] log-mel {nm} ...", flush=True)
        res[f"logmel_{nm}_5class"] = run_logmel(nm, k)
        print(f"  rec-level {res[f'logmel_{nm}_5class']['recording_level_segacc']['mean']:.3f} | "
              f"seg-level {res[f'logmel_{nm}_5class']['segment_level_segacc']['mean']:.3f}", flush=True)

    # verdict
    psd = res["psd_mlp_5class"]
    inflation = psd["segment_level_segacc"]["mean"] - psd["recording_level_segacc"]["mean"]
    res["leakage_inflation_psd_mlp"] = round(inflation, 3)
    res["verdict"] = (
        f"Segment-level (leaky) split inflates PSD-MLP balanced accuracy by "
        f"{inflation:+.3f} over the recording-level (honest) split. If large and positive, "
        f"the IARA-paper 0.675 is consistent with segment-level leakage, and our recording-level "
        f"~0.41-0.45 is the honest leakage-free correction.")
    (ROOT / "iara_leakage_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[leak] VERDICT: segment-level leakage inflates by {inflation:+.3f}", flush=True)
    print(f"[leak] done {time.time()-t0:.0f}s -> iara_leakage_results.json", flush=True)


if __name__ == "__main__":
    main()
