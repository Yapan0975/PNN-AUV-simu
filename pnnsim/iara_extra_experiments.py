"""
iara_extra_experiments.py -- round-5 self-review Bucket B:
  (1) ResNet at 15 seeds (equal power with the n=15 PAT-vs-reservoir test),
      5-class AND 4-class (dropping the acoustically-incoherent 'Special Craft'),
      on the cached 4 s log-mel features -> R1-W2 + R2-W5.
  (2) Window-length sweep: re-extract log-mel at 8 s and 16 s windows and re-train
      the ResNet 5-class -> tests whether the ceiling is window-limited (R1-W5/R2-W1).
Checkpoints after every block. Run:  py -u iara_extra_experiments.py
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import iara_dataset as iara
import iara_logmel as ilm
import models_bench as mb
from phase0_iara import rec_balanced_acc

torch.set_num_threads(16)
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
XLSX = str(DATA / "iara.xlsx")
LM4 = ROOT / "iara_logmel_features.npz"          # the cached 4 s features
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
EPOCHS = 30
OUT = ROOT / "iara_extra_experiments_results.json"
res = {"note": "round-5 review Bucket B: ResNet 15-seed (5/4-class) + window sweep"}


def resnet_acc(X, y, rec, k, seeds):
    n_mels, T = X.shape[2], X.shape[3]
    accs = []
    for s in seeds:
        torch.manual_seed(s)
        tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=s)
        Xtr, ytr = X[tr], y[tr]
        Xte, yte, rec_te = X[te], y[te], rec[te]      # materialize test subset (avoids bool-mask slicing)
        m = mb.build("resnet", n_mels, T, k)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(Xtr)
        for _ in range(EPOCHS):
            perm = torch.randperm(n); m.train()
            for i in range(0, n, 128):
                idx = perm[i:i + 128]
                opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            outs = []
            for i in range(0, len(Xte), 256):
                outs.append(torch.softmax(m(Xte[i:i + 256]), 1))
            prob = torch.cat(outs, 0).numpy()
        accs.append(rec_balanced_acc(prob, yte, rec_te, k))
    a = np.array(accs)
    lo, hi = np.percentile(a, [2.5, 97.5]) if len(a) >= 8 else (a.min(), a.max())
    return {"mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)], "n_seeds": len(seeds),
            "vals": [round(float(v), 3) for v in a]}


def main():
    t0 = time.time()
    # ---- (1) ResNet 15-seed on cached 4 s features ----
    d = np.load(LM4, allow_pickle=True)
    X = torch.from_numpy(d["X"]); y = torch.from_numpy(d["y"]).long(); rec = d["rec"]
    seeds15 = list(range(15))
    print(f"[extra] ResNet 15-seed, 5-class (cached 4 s, X{tuple(X.shape)})", flush=True)
    res["resnet_15seed_5class"] = resnet_acc(X, y, rec, 5, seeds15)
    print("   5-class:", res["resnet_15seed_5class"], f"[{time.time()-t0:.0f}s]", flush=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # 4-class: drop 'Special Craft' (label 4)
    keep = (y <= 3)
    Xk, yk, reck = X[keep], y[keep], rec[keep.numpy()]
    print("[extra] ResNet 15-seed, 4-class (drop Special Craft)", flush=True)
    res["resnet_15seed_4class_no_special"] = resnet_acc(Xk, yk, reck, 4, seeds15)
    print("   4-class:", res["resnet_15seed_4class_no_special"], f"[{time.time()-t0:.0f}s]", flush=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ---- (2) window-length sweep: re-extract 8 s and 16 s, ResNet 5-class (5 seeds) ----
    res["window_sweep_5class"] = {"reference_4s": res["resnet_15seed_5class"]["mean"]}
    for seg, mpr, tag in [(8.0, 6, "w8"), (16.0, 3, "w16")]:
        cache = ROOT / f"iara_logmel_{tag}.npz"
        print(f"[extra] window {seg:.0f}s: extracting (cache {cache.name})", flush=True)
        Xw, yw, recw, _ = ilm.load_iara_logmel(DATA, XLSX, CLASSES, seg_seconds=seg,
                                               max_per_rec=mpr, recs_per_class=45, seed=0,
                                               cache=str(cache), verbose=False)
        r = resnet_acc(Xw, yw, recw, 5, [0, 1, 2, 3, 4])
        res["window_sweep_5class"][tag] = {"seg_seconds": seg, "max_per_rec": mpr,
                                           "T_frames": int(Xw.shape[3]), **r}
        print(f"   {seg:.0f}s 5-class: {r['mean']}+/-{r['std']}  [{time.time()-t0:.0f}s]", flush=True)
        OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n[extra] DONE in {time.time()-t0:.0f}s -> {OUT.name}")
    print("  ResNet 5-cls 15-seed:", res["resnet_15seed_5class"]["mean"], res["resnet_15seed_5class"]["ci95"])
    print("  ResNet 4-cls (no special):", res["resnet_15seed_4class_no_special"]["mean"])
    print("  window 4/8/16 s:", res["resnet_15seed_5class"]["mean"],
          res["window_sweep_5class"].get("w8", {}).get("mean"),
          res["window_sweep_5class"].get("w16", {}).get("mean"))


if __name__ == "__main__":
    main()
