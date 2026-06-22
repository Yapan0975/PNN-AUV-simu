"""
iara_subset_robustness.py -- external-review (round 3) #4: is the low leakage-free
5-class ceiling (ResNet ~0.46, band-limited path ~0.40) a SUBSET ARTIFACT, or a
property of the IARA task? We re-test it on recording draws OTHER than the single
balanced 45/class subset (seed 0) used in the paper:

  * three ALTERNATE balanced draws  (45/class, subset-draw seeds 1,2,3)
  * one larger NATURAL-IMBALANCE draw (up to 150/class -> Tug/Tanker all available,
    others capped; the archive's own class proportions, not a balanced subset)

For every draw we re-extract features from the raw zips and run BOTH ends on the
SAME leakage-free by-recording split:
  * digital log-mel ResNet  (the strong-baseline "ceiling")
  * resonator-band digital MLP (the band-limited path the analog front end sees;
    the paper's own finding is that training the physics adds nothing over a
    trained readout, so this MLP-on-band-powers is the faithful analog proxy)

If the ceiling and the digital-vs-band gap stay stable across independent draws and
under natural imbalance, the "you cherry-picked a hard subset to manufacture a low
ceiling" objection fails.

Run:  py -u iara_subset_robustness.py   ->  iara_subset_robustness_results.json
"""
from __future__ import annotations
import json, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import iara_dataset as iara
import iara_logmel as ilm
import models_bench as mb
import core
from phase0_iara import rec_balanced_acc, _probs

torch.set_num_threads(16)
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
XLSX = str(DATA / "iara.xlsx")
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
K = 5
TRAIN_SEEDS = [0, 1, 2]          # split + init seeds per draw (3 for speed)
EPOCHS_RESNET = 25

# (tag, recs_per_class, draw_seed)
DRAWS = [
    ("bal_seed0", 45, 0),        # == the paper's subset (sanity reproduction)
    ("bal_seed1", 45, 1),
    ("bal_seed2", 45, 2),
    ("bal_seed3", 45, 3),
    ("imbalanced_150", 150, 0),  # natural imbalance, ~3x larger
]


def train_resnet(X, y, rec, seeds):
    """Digital log-mel ResNet, 5-class, recording-level balanced accuracy."""
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
        with torch.no_grad():
            prob = torch.softmax(m(Xte), dim=1).numpy()
        accs.append(rec_balanced_acc(prob, yte, rec_te, K))
    return accs


def train_band_mlp(X, y, rec, seeds, h=64, epochs=200, lr=3e-3, wd=1e-4):
    """Resonator-band digital MLP (1 hidden ReLU + LayerNorm) = the analog proxy."""
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
    t0 = time.time()
    cfg = core.Config()
    out = {"task": "subset robustness of the leakage-free 5-class ceiling (external review r3 #4)",
           "metric": "recording-level balanced accuracy, mean+/-std over 3 train/split seeds",
           "paper_subset_reference": {"resnet_5cls": 0.460, "band_mlp_5cls": 0.429,
                                      "analog_pnn_pat_5cls": 0.403},
           "draws": {}}
    ckpt = ROOT / "iara_subset_robustness_results.json"
    for tag, rpc, dseed in DRAWS:
        lm_cache = ROOT / f"iara_lm_robust_{tag}.npz"
        rb_cache = ROOT / f"iara_rb_robust_{tag}.npz"
        print(f"\n[robust] draw '{tag}' (recs_per_class={rpc}, draw_seed={dseed})", flush=True)
        Xlm, ylm, reclm, _ = ilm.load_iara_logmel(DATA, XLSX, CLASSES,
                                                   recs_per_class=rpc, seed=dseed,
                                                   cache=str(lm_cache), verbose=False)
        Xrb, yrb, recrb, _ = iara.load_iara(cfg, DATA, XLSX, CLASSES,
                                            recs_per_class=rpc, seed=dseed,
                                            cache=str(rb_cache), verbose=False)
        # per-class recording counts (from the log-mel draw; same selection)
        rec_first_cls = {}
        for ridx, rr in enumerate(reclm):
            rec_first_cls.setdefault(int(rr), int(ylm[ridx]))
        cls_counts = Counter(CLASSES[c] for c in rec_first_cls.values())
        n_rec = len(rec_first_cls)
        print(f"  n_rec={n_rec}  class_counts={dict(cls_counts)}", flush=True)

        res_acc = train_resnet(Xlm, ylm, reclm, TRAIN_SEEDS)
        band_acc = train_band_mlp(Xrb, yrb, recrb, TRAIN_SEEDS)
        rec5 = float(np.mean(res_acc)); rec5s = float(np.std(res_acc))
        bnd5 = float(np.mean(band_acc)); bnd5s = float(np.std(band_acc))
        out["draws"][tag] = {
            "recs_per_class_cap": rpc, "draw_seed": dseed, "n_recordings": n_rec,
            "class_counts": dict(cls_counts),
            "resnet_5cls_mean": round(rec5, 3), "resnet_5cls_std": round(rec5s, 3),
            "band_mlp_5cls_mean": round(bnd5, 3), "band_mlp_5cls_std": round(bnd5s, 3),
            "gap_resnet_minus_band": round(rec5 - bnd5, 3),
            "resnet_vals": [round(a, 3) for a in res_acc],
            "band_vals": [round(a, 3) for a in band_acc],
        }
        print(f"  ResNet 5-cls {rec5:.3f}+/-{rec5s:.3f}   band-MLP {bnd5:.3f}+/-{bnd5s:.3f}"
              f"   gap {rec5-bnd5:+.3f}   [{time.time()-t0:.0f}s]", flush=True)
        ckpt.write_text(json.dumps(out, indent=2), encoding="utf-8")   # checkpoint each draw

    # summary across the balanced draws
    bal = [v for t, v in out["draws"].items() if t.startswith("bal_")]
    out["balanced_draw_summary"] = {
        "resnet_5cls_across_draws_mean": round(float(np.mean([b["resnet_5cls_mean"] for b in bal])), 3),
        "resnet_5cls_across_draws_std": round(float(np.std([b["resnet_5cls_mean"] for b in bal])), 3),
        "band_5cls_across_draws_mean": round(float(np.mean([b["band_mlp_5cls_mean"] for b in bal])), 3),
        "band_5cls_across_draws_std": round(float(np.std([b["band_mlp_5cls_mean"] for b in bal])), 3),
        "n_balanced_draws": len(bal),
    }
    ckpt.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[robust] DONE in {time.time()-t0:.0f}s")
    print("  balanced-draw ResNet ceiling:", out["balanced_draw_summary"]["resnet_5cls_across_draws_mean"],
          "+/-", out["balanced_draw_summary"]["resnet_5cls_across_draws_std"])
    print("  -> iara_subset_robustness_results.json")


if __name__ == "__main__":
    main()
