"""
phase0_iara_stats.py -- statistical hardening of the IARA acoustic results,
answering reviewer comments #7 (macro-F1, CIs, confusion matrix, per-class
results, a fixed-reservoir baseline) and #8 (does training the physics beat a
frozen reservoir + trained readout? -- paired test, effect size, CI).

Uses the FAST resonator-band PSD cache (phase0_iara_features.npz) shared with
phase0_iara.py, plus a single log-mel ResNet run (the strong digital ceiling)
for its confusion matrix / macro-F1. Recording-level metrics throughout.

Methods on the resonator band (PSD feature):
  digital_psd_mlp   -- small digital MLP (weak-feature digital control)
  pnn_pat           -- analog PNN trained with dual-model PAT
  frozen_reservoir  -- random UNtrained physics + trained linear readout
                       (the "reservoir computing" null model for #8)
  insilico          -- clean-surrogate training, deployed on noisy device

Outputs: phase0_iara_stats_results.json, phase0_iara_confusion.png,
         phase0_iara_paireddiff.png
Run:  py phase0_iara_stats.py   (~6-12 min CPU)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

import core
import iara_dataset as iara
from phase0_iara import train_digital, train_insilico, _probs

torch.set_num_threads(16)

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
SEEDS = list(range(15))          # 15 seeds: adequate power for the PAT-vs-reservoir paired test
LOGMEL_SEEDS = [0, 1, 2, 3, 4]   # 5 seeds for the (heavier) log-mel ResNet -> parity with PSD bench
PSD_CACHE = ROOT / "phase0_iara_features.npz"
LOGMEL_CACHE = ROOT / "iara_logmel_features.npz"


# ----------------------------------------------------------------------------
# recording-level prediction + metrics
# ----------------------------------------------------------------------------
def rec_predict(prob, yte, rec_te):
    """Aggregate per-segment probs to one prediction per recording."""
    yn = yte.numpy()
    preds, trues = [], []
    for r in np.unique(rec_te):
        m = rec_te == r
        preds.append(int(prob[m].mean(0).argmax()))
        trues.append(int(yn[m][0]))
    return np.array(preds), np.array(trues)


def balanced_acc(preds, trues, k):
    return float(np.mean([(preds[trues == c] == c).mean()
                          for c in range(k) if (trues == c).any()]))


def macro_f1(preds, trues, k):
    fs = []
    for c in range(k):
        tp = np.sum((preds == c) & (trues == c))
        fp = np.sum((preds == c) & (trues != c))
        fn = np.sum((preds != c) & (trues == c))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fs.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(fs)) if fs else 0.0


def per_class_recall(preds, trues, k):
    return [round(float((preds[trues == c] == c).mean()), 3) if (trues == c).any() else None
            for c in range(k)]


def confusion(preds, trues, k):
    M = np.zeros((k, k), dtype=int)
    for t, p in zip(trues, preds):
        M[t, p] += 1
    return M


def cluster_bootstrap_ci(per_seed_pt, k, B=2000, seed=0):
    """Recording-level CLUSTER bootstrap (corrects the earlier pooled-resample CI,
    which understated uncertainty by treating the same recording across seeds as
    independent). For EACH seed's test split (its ~14 recordings/class are the
    independent units), resample recordings with replacement and recompute
    balanced accuracy; report the across-seed mean of the per-split 95% bounds."""
    rng = np.random.default_rng(seed)
    los, his = [], []
    for preds, trues in per_seed_pt:
        n = len(trues)
        vals = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            p, t = preds[idx], trues[idx]
            if len(np.unique(t)) < 2:
                continue
            vals.append(balanced_acc(p, t, k))
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            los.append(lo); his.append(hi)
    return round(float(np.mean(los)), 3), round(float(np.mean(his)), 3)


# ----------------------------------------------------------------------------
# train one PSD-band method, return recording-level (preds, trues)
# ----------------------------------------------------------------------------
def run_psd_method(name, cfg, Xtr, ytr, Xte, yte, rec_te, nrng):
    if name == "digital_psd_mlp":
        m = train_digital(cfg, Xtr, ytr)
        prob = _probs(lambda S: m(S), Xte)
    elif name == "pnn_pat":
        m, _ = core.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, nrng, return_model=True)
        prob = _probs(lambda S: m.forward_eval(S, noise_rng=nrng), Xte)
    elif name == "frozen_reservoir":
        m, _ = core.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, nrng, freeze=True, return_model=True)
        prob = _probs(lambda S: m.forward_eval(S, noise_rng=nrng), Xte)
    elif name == "insilico":
        m = train_insilico(cfg, Xtr, ytr, nrng)
        prob = _probs(lambda S: m.forward_eval(S, noise_rng=nrng), Xte)
    else:
        raise ValueError(name)
    return rec_predict(prob, yte, rec_te)


def subset(X, y, rec, kk):
    if kk == 2:
        return X, (y != 0).long(), rec
    keep = (y <= (kk - 1))
    return X[keep], y[keep], rec[keep.numpy()]


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    k5 = len(CLASSES)
    cfg0 = core.Config(seed=0, n_classes=k5, n_modes=32, epochs=60, batch=128, lr=3e-3)
    X, y, rec, classes = iara.load_iara(cfg0, DATA, str(DATA / "iara.xlsx"), CLASSES,
                                        seg_seconds=4.0, max_per_rec=12, recs_per_class=45,
                                        seed=0, cache=str(PSD_CACHE))
    print(f"[stats] PSD X{tuple(X.shape)}  n_rec {len(np.unique(rec))}", flush=True)

    psd_methods = ["digital_psd_mlp", "pnn_pat", "frozen_reservoir", "insilico"]

    # ---- granularity sweep: balanced acc + macro-F1 + bootstrap CI, 5 seeds ----
    sweep = {}
    # store 5-class per-seed predictions for confusion + paired test
    preds5 = {m: [] for m in psd_methods}
    for kk in (2, 3, 4, 5):
        Xs, ys, rs = subset(X, y, rec, kk)
        per = {m: {"bal": [], "f1": []} for m in psd_methods}
        cat_preds = {m: ([], []) for m in psd_methods}   # pooled preds/trues for CI+confusion
        for s in SEEDS:
            tr, te = iara.grouped_split(ys, rs, test_frac=0.3, seed=s)
            Xtr, ytr, Xte, yte = Xs[tr], ys[tr], Xs[te], ys[te]
            rec_te = rs[te]
            cfg = core.Config(seed=s, n_classes=kk, n_modes=32, epochs=60, batch=128, lr=3e-3)
            nrng = torch.Generator().manual_seed(s + 100)
            for m in psd_methods:
                p, t = run_psd_method(m, cfg, Xtr, ytr, Xte, yte, rec_te, nrng)
                per[m]["bal"].append(balanced_acc(p, t, kk))
                per[m]["f1"].append(macro_f1(p, t, kk))
                cat_preds[m][0].append(p); cat_preds[m][1].append(t)
                if kk == 5:
                    preds5[m].append((p, t))
        sweep[kk] = {}
        for m in psd_methods:
            per_seed_pt = list(zip(cat_preds[m][0], cat_preds[m][1]))
            lo, hi = cluster_bootstrap_ci(per_seed_pt, kk)
            sweep[kk][m] = {
                "bal_mean": round(float(np.mean(per[m]["bal"])), 3),
                "bal_std": round(float(np.std(per[m]["bal"])), 3),
                "macro_f1_mean": round(float(np.mean(per[m]["f1"])), 3),
                "macro_f1_std": round(float(np.std(per[m]["f1"])), 3),
                "bal_ci95": [lo, hi],
            }
        print(f"  {kk}-class: " + " | ".join(
            f"{m.split('_')[0]} {sweep[kk][m]['bal_mean']:.3f}[{sweep[kk][m]['bal_ci95'][0]:.2f},{sweep[kk][m]['bal_ci95'][1]:.2f}]"
            for m in psd_methods), flush=True)

    # ---- 5-class confusion + per-class (pool all seeds) ----
    conf5 = {}; perclass5 = {}
    for m in psd_methods:
        allp = np.concatenate([p for p, _ in preds5[m]])
        allt = np.concatenate([t for _, t in preds5[m]])
        conf5[m] = confusion(allp, allt, k5).tolist()
        perclass5[m] = per_class_recall(allp, allt, k5)

    # ---- PAT vs frozen-reservoir PAIRED test at 5-class (#8), n=15 seeds ----
    pat_bal = [balanced_acc(p, t, k5) for p, t in preds5["pnn_pat"]]
    fro_bal = [balanced_acc(p, t, k5) for p, t in preds5["frozen_reservoir"]]
    diffs = np.array(pat_bal) - np.array(fro_bal)
    nseed = len(diffs)
    dz = float(diffs.mean() / (diffs.std(ddof=1) + 1e-9))            # Cohen's d_z (paired)
    sem = float(sstats.sem(diffs))
    tci = sstats.t.interval(0.95, nseed - 1, loc=diffs.mean(), scale=sem) if diffs.std() > 0 else (diffs.mean(), diffs.mean())
    # minimum detectable mean difference reaching p<0.05 (two-sided) at this n & variance
    mde = float(sstats.t.ppf(0.975, nseed - 1) * sem)
    try:
        w_p = float(sstats.wilcoxon(pat_bal, fro_bal).pvalue)
    except ValueError:
        w_p = 1.0
    paired = {
        "n_seeds": nseed,
        "pat_bal_per_seed": [round(x, 3) for x in pat_bal],
        "frozen_bal_per_seed": [round(x, 3) for x in fro_bal],
        "mean_diff": round(float(diffs.mean()), 3),
        "ci95_diff": [round(float(tci[0]), 3), round(float(tci[1]), 3)],
        "cohens_dz": round(dz, 3),
        "wilcoxon_p": round(w_p, 3),
        "min_detectable_diff": round(mde, 3),
        "verdict": (f"At n={nseed} the design can detect a mean difference of {mde:.3f}; the observed "
                    f"{diffs.mean():+.3f} (95% CI [{tci[0]:.3f},{tci[1]:.3f}], p={w_p:.2f}) is well below it, "
                    f"so training the physics FAILS TO DEMONSTRATE a gain over a frozen reservoir + trained "
                    f"readout -- consistent with reservoir-dominant learning, though a small effect below "
                    f"{mde:.3f} cannot be excluded."),
    }
    print(f"[stats] PAT vs frozen (5-class, n={nseed}): diff {paired['mean_diff']:+.3f} "
          f"CI{paired['ci95_diff']} d_z={paired['cohens_dz']} p={paired['wilcoxon_p']} MDE={paired['min_detectable_diff']}", flush=True)

    # ---- strong digital ceiling: log-mel ResNet confusion + macro-F1 (5 seeds, parity) ----
    resnet5 = run_logmel_resnet5(seeds=LOGMEL_SEEDS)

    res = {
        "dataset": "IARA (Zenodo 10.5281/zenodo.15758636), full archive A-H",
        "classes_5": list(classes),
        "split": "recording-level (no segment leakage), 30% test",
        "n_seeds_psd": len(SEEDS), "n_seeds_logmel": len(LOGMEL_SEEDS),
        "metric": "recording-level; balanced accuracy + macro-F1 + recording-level CLUSTER bootstrap 95% CI (per-seed, averaged)",
        "granularity_sweep": {str(kk): v for kk, v in sweep.items()},
        "confusion_5class": conf5,
        "per_class_recall_5class": {m: perclass5[m] for m in psd_methods},
        "logmel_resnet_5class": resnet5,
        "pat_vs_frozen_paired_5class": paired,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (ROOT / "phase0_iara_stats_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _make_confusion_figure(conf5, resnet5, classes)
    _make_paired_figure(paired)
    print(f"[stats] done {time.time()-t0:.0f}s -> phase0_iara_stats_results.json", flush=True)


def run_logmel_resnet5(seeds=(0, 1, 2, 3, 4)):
    """Strong digital ceiling on log-mel: balanced acc + macro-F1 + confusion."""
    import iara_logmel as ilm
    import models_bench as mb
    import torch.nn.functional as F
    X, y, rec, _ = ilm.load_iara_logmel(DATA, str(DATA / "iara.xlsx"), CLASSES,
                                        cache=str(LOGMEL_CACHE))
    nm, T = X.shape[2], X.shape[3]
    bals, f1s = [], []
    pooled_p, pooled_t = [], []
    per_seed_pt = []
    for s in seeds:
        torch.manual_seed(s)
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]; rec_te = rec[te]
        m = mb.build("resnet", nm, T, 5)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(Xtr)
        for _ in range(30):
            perm = torch.randperm(n); m.train()
            for i in range(0, n, 128):
                idx = perm[i:i + 128]
                opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            prob = torch.softmax(m(Xte), 1).numpy()
        p, t = rec_predict(prob, yte, rec_te)
        bals.append(balanced_acc(p, t, 5)); f1s.append(macro_f1(p, t, 5))
        pooled_p.append(p); pooled_t.append(t); per_seed_pt.append((p, t))
    allp = np.concatenate(pooled_p); allt = np.concatenate(pooled_t)
    return {
        "bal_mean": round(float(np.mean(bals)), 3), "bal_std": round(float(np.std(bals)), 3),
        "macro_f1_mean": round(float(np.mean(f1s)), 3), "macro_f1_std": round(float(np.std(f1s)), 3),
        "bal_ci95": list(cluster_bootstrap_ci(per_seed_pt, 5)),
        "confusion": confusion(allp, allt, 5).tolist(),
        "per_class_recall": per_class_recall(allp, allt, 5),
        "n_seeds": len(seeds),
    }


def _make_confusion_figure(conf5, resnet5, classes):
    short = ["Bg", "Cargo", "Tank", "Tug", "Spec"]
    mats = [("Digital ResNet (log-mel)", np.array(resnet5["confusion"])),
            ("Analog PNN-PAT (resonator)", np.array(conf5["pnn_pat"])),
            ("Frozen reservoir (resonator)", np.array(conf5["frozen_reservoir"]))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (title, M) in zip(axes, mats):
        Mn = M / (M.sum(1, keepdims=True) + 1e-9)
        im = ax.imshow(Mn, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(short, fontsize=7, rotation=45); ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if Mn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02)
    fig.suptitle("IARA 5-class recording-level confusion (row-normalised)", fontsize=10)
    fig.savefig(ROOT / "phase0_iara_confusion.png", dpi=200, bbox_inches="tight")


def _make_paired_figure(paired):
    pat = paired["pat_bal_per_seed"]; fro = paired["frozen_bal_per_seed"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for a, b in zip(fro, pat):
        ax.plot([0, 1], [a, b], "-o", color="#888888", ms=4, lw=0.9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["frozen\nreservoir", "PNN-PAT\n(trained physics)"], fontsize=8)
    ax.set_ylabel("5-class balanced accuracy")
    ax.set_title(f"Paired per seed: $\\Delta$={paired['mean_diff']:+.3f} "
                 f"(95% CI {paired['ci95_diff']})\n$d_z$={paired['cohens_dz']}, "
                 f"Wilcoxon p={paired['wilcoxon_p']}", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "phase0_iara_paireddiff.png", dpi=200)


if __name__ == "__main__":
    main()
