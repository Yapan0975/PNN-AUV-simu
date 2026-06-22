"""
phase0_iara.py -- Phase 0 on REAL IARA hydrophone recordings (full archive A-H).

Real underwater target classification on the IARA archive
(Zenodo 10.5281/zenodo.15758636), classes Background / Cargo / Tanker / Tug /
Special Craft, recording-level train/test split (no segment leakage), 5 seeds.

Honest findings reported by this script:
  * the analog PNN-PAT front end RUNS on real ocean audio and TRACKS the digital
    baseline across task granularity, but with a ~0.05-0.10 balanced-accuracy
    COST on real spectra (absent in the synthetic stress test; not closed by
    more resonators -> a device-noise penalty, see iara_task_sweep.py);
  * fine-grained 5-class classification is hard for ALL methods (incl. the
    digital MLP) with this minimal 10-1000 Hz front end;
  * the in-silico>PAT discrepancy seen on the F+G+H subset did NOT survive
    5 seeds (small-sample artifact: PAT ~ in-silico ~ frozen at 5-class).
Power advantage is the same device-anchored ~94x as synthetic Phase 0
(data-independent), not re-derived here.

Run:  py phase0_iara.py
Outputs: phase0_iara_results.json, phase0_iara_figure.png
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

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
SEEDS = [0, 1, 2, 3, 4]
CACHE = ROOT / "phase0_iara_features.npz"


def balanced_acc(pred, y, k):
    r = [((pred[y == c] == c).float().mean().item()) for c in range(k) if (y == c).any()]
    return float(np.mean(r))


def _probs(fn, Xte):
    with torch.no_grad():
        return torch.softmax(fn(Xte), dim=1).numpy()


def rec_balanced_acc(prob, yte, rec_te, k):
    """Recording-level balanced accuracy: average per-segment class probabilities
    within each recording, then argmax (the operational ship-classification
    metric -- classify the vessel, not each 4 s window)."""
    yn = yte.numpy()
    preds, trues = [], []
    for r in np.unique(rec_te):
        m = rec_te == r
        preds.append(int(prob[m].mean(0).argmax()))
        trues.append(int(yn[m][0]))
    preds, trues = np.array(preds), np.array(trues)
    return float(np.mean([(preds[trues == c] == c).mean() for c in range(k) if (trues == c).any()]))


def train_digital(cfg, Xtr, ytr):
    torch.manual_seed(cfg.seed)
    m = core.DigitalBaseline(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
    return m


def train_insilico(cfg, Xtr, ytr, noise_rng):
    torch.manual_seed(cfg.seed)
    m = core.PNNClassifier(cfg, freeze_physics=False)
    m.array.set_device_instance(noise_rng)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=cfg.lr)
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad(); F.cross_entropy(m.forward_clean(Xtr[idx]), ytr[idx]).backward(); opt.step()
    return m


def _eval_models(X, y, rec, k, seeds, with_ablations=False):
    """Recording-level balanced-accuracy mean/std over seeds for the digital
    baseline and PNN-PAT (and frozen-physics + in-silico if requested)."""
    keys = ["digital", "pnn_pat"] + (["frozen", "insilico"] if with_ablations else [])
    out = {m: [] for m in keys}
    for s in seeds:
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        rec_te = rec[te]
        cfg = core.Config(seed=s, n_classes=k, n_modes=32, epochs=60, batch=128, lr=3e-3)
        nrng = torch.Generator().manual_seed(s + 100)
        dig = train_digital(cfg, Xtr, ytr)
        mp, _ = core.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, nrng, return_model=True)
        out["digital"].append(rec_balanced_acc(_probs(lambda S: dig(S), Xte), yte, rec_te, k))
        out["pnn_pat"].append(rec_balanced_acc(_probs(lambda S: mp.forward_eval(S, noise_rng=nrng), Xte), yte, rec_te, k))
        if with_ablations:
            mf, _ = core.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, nrng, freeze=True, return_model=True)
            mi = train_insilico(cfg, Xtr, ytr, nrng)
            out["frozen"].append(rec_balanced_acc(_probs(lambda S: mf.forward_eval(S, noise_rng=nrng), Xte), yte, rec_te, k))
            out["insilico"].append(rec_balanced_acc(_probs(lambda S: mi.forward_eval(S, noise_rng=nrng), Xte), yte, rec_te, k))
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "vals": [round(x, 4) for x in v]}
            for m, v in out.items()}


def _make_figure(sweep, k5):
    ks = [2, 3, 4, 5]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax1.errorbar(ks, [sweep[kk]["digital"]["mean"] for kk in ks],
                 yerr=[sweep[kk]["digital"]["std"] for kk in ks],
                 marker="s", color="#888888", capsize=3, label="Digital baseline")
    ax1.errorbar(ks, [sweep[kk]["pnn_pat"]["mean"] for kk in ks],
                 yerr=[sweep[kk]["pnn_pat"]["std"] for kk in ks],
                 marker="o", color="#2c6fbb", capsize=3, label="PNN-PAT (noisy device)")
    ax1.plot(ks, [1.0 / kk for kk in ks], ls=":", color="k", lw=0.9, label="chance")
    ax1.set_xticks(ks); ax1.set_xlabel("number of classes")
    ax1.set_ylabel("recording-level balanced accuracy"); ax1.set_ylim(0, 0.85)
    ax1.set_title("Real IARA: PNN-PAT tracks the digital baseline\nwith an honest accuracy cost", fontsize=9)
    ax1.grid(alpha=0.25); ax1.legend(fontsize=7)
    order = ["digital", "pnn_pat", "frozen", "insilico"]
    labels = ["Digital", "PNN-PAT", "Frozen\nphys.", "In-silico"]
    ax2.bar(labels, [sweep[5][m]["mean"] for m in order], yerr=[sweep[5][m]["std"] for m in order],
            capsize=3, color=["#888888", "#2c6fbb", "#cc5500", "#bbbbbb"])
    ax2.axhline(1.0 / k5, ls=":", color="k", lw=0.8)
    ax2.set_ylabel("balanced accuracy"); ax2.set_ylim(0, 0.6)
    ax2.set_title("5-class: PAT $\\approx$ in-silico $\\approx$ frozen\n(in-silico>PAT artifact debunked)", fontsize=9)
    ax2.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(ROOT / "phase0_iara_figure.png", dpi=200)


def main():
    t0 = time.time()
    k = len(CLASSES)
    cfg0 = core.Config(seed=0, n_classes=k, n_modes=32, epochs=60, batch=128, lr=3e-3)
    X, y, rec, classes = iara.load_iara(cfg0, DATA, str(DATA / "iara.xlsx"), CLASSES,
                                        seg_seconds=4.0, max_per_rec=12, recs_per_class=45,
                                        seed=0, cache=str(CACHE))
    counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    print(f"[iara] total {len(X)} segments  counts {counts}")

    # task-granularity sweep (binary -> 5-class), recording-level, 5 seeds
    sweep = {}
    for kk, name in [(2, "ship vs background"), (3, "bg/cargo/tanker"),
                     (4, "bg/cargo/tanker/tug"), (5, "5-class (all)")]:
        if kk == 2:
            Xs, ys, rs = X, (y != 0).long(), rec
        else:
            keep = (y <= (kk - 1))
            Xs, ys, rs = X[keep], y[keep], rec[keep.numpy()]
        sweep[kk] = {"name": name, **_eval_models(Xs, ys, rs, kk, SEEDS, with_ablations=(kk == 5))}
        print(f"  {kk}-class ({name}): digital {sweep[kk]['digital']['mean']:.3f} | "
              f"PNN-PAT {sweep[kk]['pnn_pat']['mean']:.3f}+/-{sweep[kk]['pnn_pat']['std']:.3f}")

    res = {
        "dataset": "IARA (Zenodo 10.5281/zenodo.15758636), full archive A-H, real 128 kHz hydrophone audio",
        "classes_5": list(classes), "class_counts": counts,
        "n_segments": int(len(X)), "n_seeds": len(SEEDS),
        "split": "recording-level (no segment leakage), 30% test",
        "metric": "recording-level balanced accuracy (segments aggregated per recording)",
        "granularity_sweep": {str(kk): v for kk, v in sweep.items()},
        "pnn_minus_digital_binary": sweep[2]["pnn_pat"]["mean"] - sweep[2]["digital"]["mean"],
        "pat_minus_insilico_5class": sweep[5]["pnn_pat"]["mean"] - sweep[5]["insilico"]["mean"],
        "note": "HONEST: analog PNN tracks the digital baseline but with a ~0.05-0.10 cost on "
                "real spectra (absent synthetically); not closed by more resonators (device-noise "
                "penalty). Fine-grained classification hard for ALL methods with this minimal "
                "10-1000 Hz front end. Power advantage = same device-model ~94x (data-independent). "
                "in-silico>PAT on F+G+H subset did NOT survive 5 seeds (artifact).",
        "wall_clock_s": time.time() - t0,
    }
    (ROOT / "phase0_iara_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _make_figure(sweep, k)

    print("\n===== REAL IARA SUMMARY (recording-level, 5 seeds) =====")
    for kk in (2, 3, 4, 5):
        d, p = sweep[kk]["digital"], sweep[kk]["pnn_pat"]
        print(f"{kk}-class chance {1/kk:.2f} | digital {d['mean']:.3f} | "
              f"PNN-PAT {p['mean']:.3f}+/-{p['std']:.3f} | gap {d['mean']-p['mean']:+.3f}")
    print(f"5-class: PAT {sweep[5]['pnn_pat']['mean']:.3f} ~ in-silico {sweep[5]['insilico']['mean']:.3f} "
          f"~ frozen {sweep[5]['frozen']['mean']:.3f}  (anomaly debunked)")
    print(f"wall clock {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
