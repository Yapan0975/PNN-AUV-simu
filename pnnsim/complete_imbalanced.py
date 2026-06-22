"""
complete_imbalanced.py -- the imbalanced draw of iara_subset_robustness.py died
during training (features were already cached). This finishes just that draw from
the cached features (instant load, no re-extraction), with memory-conservative
batched evaluation, and merges the row + summary into the existing results JSON.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import iara_dataset as iara
import models_bench as mb
from phase0_iara import rec_balanced_acc, _probs

torch.set_num_threads(8)
ROOT = Path(__file__).resolve().parent
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
K = 5
TRAIN_SEEDS = [0, 1, 2]
EPOCHS_RESNET = 25


def _load(npz):
    d = np.load(npz, allow_pickle=True)
    return torch.from_numpy(d["X"]), torch.from_numpy(d["y"]).long(), d["rec"]


def _eval_batched(model, Xte, bs=256):
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            outs.append(torch.softmax(model(Xte[i:i + bs]), dim=1))
    return torch.cat(outs, 0).numpy()


def train_resnet(X, y, rec, seeds):
    n_mels, T = X.shape[2], X.shape[3]
    accs = []
    for s in seeds:
        torch.manual_seed(s)
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        rec_te = rec[te]
        m = mb.build("resnet", n_mels, T, K)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(Xtr)
        for _ in range(EPOCHS_RESNET):
            perm = torch.randperm(n); m.train()
            for i in range(0, n, 128):
                idx = perm[i:i + 128]
                opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        prob = _eval_batched(m, Xte)
        accs.append(rec_balanced_acc(prob, yte, rec_te, K))
        print(f"    resnet seed {s}: {accs[-1]:.3f}", flush=True)
    return accs


def train_band_mlp(X, y, rec, seeds, h=64, epochs=200, lr=3e-3, wd=1e-4):
    accs = []
    for s in seeds:
        torch.manual_seed(s)
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        rec_te = rec[te]
        mlp = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], h), torch.nn.ReLU(),
                                  torch.nn.LayerNorm(h), torch.nn.Linear(h, K))
        opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
        n = len(Xtr)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 128):
                idx = perm[i:i + 128]
                opt.zero_grad(); F.cross_entropy(mlp(Xtr[idx]), ytr[idx]).backward(); opt.step()
        prob = _probs(lambda S: mlp(S), Xte)
        accs.append(rec_balanced_acc(prob, yte, rec_te, K))
    return accs


def main():
    tag = "imbalanced_150"
    Xlm, ylm, reclm = _load(ROOT / f"iara_lm_robust_{tag}.npz")
    Xrb, yrb, recrb = _load(ROOT / f"iara_rb_robust_{tag}.npz")
    rec_first = {}
    for i, r in enumerate(reclm):
        rec_first.setdefault(int(r), int(ylm[i]))
    cls_counts = Counter(CLASSES[c] for c in rec_first.values())
    n_rec = len(rec_first)
    print(f"[complete] {tag}: n_rec={n_rec} counts={dict(cls_counts)}", flush=True)

    res_acc = train_resnet(Xlm, ylm, reclm, TRAIN_SEEDS)
    band_acc = train_band_mlp(Xrb, yrb, recrb, TRAIN_SEEDS)
    rec5, rec5s = float(np.mean(res_acc)), float(np.std(res_acc))
    bnd5, bnd5s = float(np.mean(band_acc)), float(np.std(band_acc))

    js = ROOT / "iara_subset_robustness_results.json"
    out = json.loads(js.read_text(encoding="utf-8"))
    out["draws"][tag] = {
        "recs_per_class_cap": 150, "draw_seed": 0, "n_recordings": n_rec,
        "class_counts": dict(cls_counts),
        "resnet_5cls_mean": round(rec5, 3), "resnet_5cls_std": round(rec5s, 3),
        "band_mlp_5cls_mean": round(bnd5, 3), "band_mlp_5cls_std": round(bnd5s, 3),
        "gap_resnet_minus_band": round(rec5 - bnd5, 3),
        "resnet_vals": [round(a, 3) for a in res_acc],
        "band_vals": [round(a, 3) for a in band_acc],
    }
    bal = [v for t, v in out["draws"].items() if t.startswith("bal_")]
    out["balanced_draw_summary"] = {
        "resnet_5cls_across_draws_mean": round(float(np.mean([b["resnet_5cls_mean"] for b in bal])), 3),
        "resnet_5cls_across_draws_std": round(float(np.std([b["resnet_5cls_mean"] for b in bal])), 3),
        "band_5cls_across_draws_mean": round(float(np.mean([b["band_mlp_5cls_mean"] for b in bal])), 3),
        "band_5cls_across_draws_std": round(float(np.std([b["band_mlp_5cls_mean"] for b in bal])), 3),
        "n_balanced_draws": len(bal),
    }
    js.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[complete] imbalanced: ResNet {rec5:.3f}+/-{rec5s:.3f}  band-MLP {bnd5:.3f}+/-{bnd5s:.3f}"
          f"  gap {rec5-bnd5:+.3f}", flush=True)
    print("  balanced ceiling:", out["balanced_draw_summary"]["resnet_5cls_across_draws_mean"],
          "+/-", out["balanced_draw_summary"]["resnet_5cls_across_draws_std"])


if __name__ == "__main__":
    main()
